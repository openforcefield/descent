"""Train against relative energies and forces."""

import typing

import datasets
import datasets.table
import pyarrow
import smee
import smee.utils
import torch

DATA_SCHEMA = pyarrow.schema(
    [
        ("id", pyarrow.string()),
        ("smiles", pyarrow.string()),
        ("coords", pyarrow.list_(pyarrow.float64())),
        ("box_vectors", pyarrow.list_(pyarrow.float64())),
        ("energy", pyarrow.list_(pyarrow.float64())),
        ("forces", pyarrow.list_(pyarrow.float64())),
    ]
)


class Entry(typing.TypedDict):
    """Represents a set of reference energies and forces."""

    id: str | None
    """An optional identifier for the entry (e.g. a run name). Defaults to ``None``."""

    smiles: str
    """The indexed SMILES description of the molecule the energies and forces were
    computed for."""

    coords: torch.Tensor
    """The coordinates [Å] the energies and forces were evaluated at with
    ``shape=(n_confs, n_particles, 3)``."""
    energy: torch.Tensor
    """The reference energies [kcal/mol] with ``shape=(n_confs,)``."""
    forces: torch.Tensor
    """The reference forces [kcal/mol/Å] with ``shape=(n_confs, n_particles, 3)``."""

    box_vectors: torch.Tensor | None
    """The box vectors [Å] for periodic systems with ``shape=(n_confs, 3, 3)``, or
    ``None`` for non-periodic systems."""


def create_dataset(entries: list[Entry]) -> datasets.Dataset:
    """Create a dataset from a list of existing entries.

    Args:
        entries: The entries to create the dataset from.

    Returns:
        The created dataset.
    """

    table = pyarrow.Table.from_pylist(
        [
            {
                "id": entry.get("id"),
                "smiles": entry["smiles"],
                "coords": torch.tensor(entry["coords"]).flatten().tolist(),
                "box_vectors": None
                if entry.get("box_vectors") is None
                else torch.tensor(entry["box_vectors"]).flatten().tolist(),
                "energy": torch.tensor(entry["energy"]).flatten().tolist(),
                "forces": torch.tensor(entry["forces"]).flatten().tolist(),
            }
            for entry in entries
        ],
        schema=DATA_SCHEMA,
    )
    # TODO: validate rows
    dataset = datasets.Dataset(datasets.table.InMemoryTable(table))
    dataset.set_format("torch")

    return dataset


def create_dataset_from_generator(
    gen_fn: typing.Callable[[], typing.Iterator[Entry]],
) -> datasets.Dataset:
    """Create a dataset from a generator function, avoiding loading all entries into
    memory at once.

    Args:
        gen_fn: A callable that returns an iterator of entries. It will be called by
            the HuggingFace datasets library and must be re-iterable (i.e. each call
            to ``gen_fn()`` should produce a fresh iterator).

    Returns:
        The created dataset.
    """

    def _gen():
        for entry in gen_fn():
            yield {
                "smiles": entry["smiles"],
                "coords": torch.tensor(entry["coords"]).flatten().tolist(),
                "energy": torch.tensor(entry["energy"]).flatten().tolist(),
                "forces": torch.tensor(entry["forces"]).flatten().tolist(),
            }

    features = datasets.Features.from_arrow_schema(DATA_SCHEMA)
    dataset = datasets.Dataset.from_generator(_gen, features=features)
    dataset.set_format("torch")

    return dataset


def extract_smiles(dataset: datasets.Dataset) -> list[str]:
    """Return a list of unique SMILES strings in the dataset.

    Args:
        dataset: The dataset to extract the SMILES strings from.

    Returns:
        The list of unique SMILES strings.
    """
    return sorted({*dataset.unique("smiles")})


def predict(
    dataset: datasets.Dataset,
    force_field: smee.TensorForceField,
    topologies: dict[str, smee.TensorTopology],
    reference: typing.Literal["mean", "min"] = "mean",
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict the relative energies [kcal/mol] and forces [kcal/mol/Å] of a dataset.

    Args:
        dataset: The dataset to predict the energies and forces of.
        force_field: The force field to use to predict the energies and forces.
        topologies: The topologies of the molecules in the dataset. Each key should be
            a fully indexed SMILES string.
        reference: The reference energy to compute the relative energies with respect
            to. This should be either the "mean" energy of all conformers, or the
            energy of the conformer with the lowest reference energy ("min").
        normalize: Whether to scale the relative energies by ``1/sqrt(n_confs_i)``
            and the forces by ``1/sqrt(n_confs_i * n_atoms_per_conf_i * 3)`` This
            is useful when wanting to compute the MSE per entry.

    Returns:
        The predicted and reference relative energies [kcal/mol] with
        ``shape=(n_confs,)``, and predicted and reference forces [kcal/mol/Å] with
        ``shape=(n_confs * n_atoms_per_conf, 3)``.
    """
    energy_ref_all, energy_pred_all = [], []
    forces_ref_all, forces_pred_all = [], []

    for entry in dataset:
        smiles = entry["smiles"]

        energy_ref = entry["energy"]
        forces_ref = entry["forces"].reshape(len(energy_ref), -1, 3)

        coords_flat = smee.utils.tensor_like(
            entry["coords"], force_field.potentials[0].parameters
        )

        coords = (
            (coords_flat.reshape(len(energy_ref), -1, 3)).detach().requires_grad_(True)
        )
        box_vectors = entry.get("box_vectors", None)

        topology = topologies[smiles]

        if box_vectors is not None:
            # smee does not support batched periodic evaluations,
            # so we loop over conformers.
            box_vectors = smee.utils.tensor_like(box_vectors, coords_flat).reshape(
                len(energy_ref), 3, 3
            )
            energy_pred = torch.cat(
                [
                    smee.compute_energy(
                        topology, force_field, coords[i], box_vectors[i]
                    )
                    for i in range(len(energy_ref))
                ]
            )
        else:
            energy_pred = smee.compute_energy(topology, force_field, coords, None)

        forces_pred = -torch.autograd.grad(
            energy_pred.sum(),
            coords,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if reference.lower() == "mean":
            energy_ref_0 = energy_ref.mean()
            energy_pred_0 = energy_pred.mean()
        elif reference.lower() == "min":
            min_idx = energy_ref.argmin()

            energy_ref_0 = energy_ref[min_idx]
            energy_pred_0 = energy_pred[min_idx]
        else:
            raise NotImplementedError(f"invalid reference energy {reference}")

        scale_energy, scale_forces = 1.0, 1.0

        if normalize:
            scale_energy = 1.0 / torch.sqrt(torch.tensor(energy_pred.numel()))
            scale_forces = 1.0 / torch.sqrt(torch.tensor(forces_pred.numel()))

        energy_ref_all.append(scale_energy * (energy_ref - energy_ref_0))
        forces_ref_all.append(scale_forces * forces_ref.reshape(-1, 3))

        energy_pred_all.append(scale_energy * (energy_pred - energy_pred_0))
        forces_pred_all.append(scale_forces * forces_pred.reshape(-1, 3))

    energy_pred_all = torch.cat(energy_pred_all)
    forces_pred_all = torch.cat(forces_pred_all)

    energy_ref_all = torch.cat(energy_ref_all)
    energy_ref_all = smee.utils.tensor_like(energy_ref_all, energy_pred_all)

    forces_ref_all = torch.cat(forces_ref_all)
    forces_ref_all = smee.utils.tensor_like(forces_ref_all, forces_pred_all)

    return (
        energy_ref_all,
        energy_pred_all,
        forces_ref_all,
        forces_pred_all,
    )
