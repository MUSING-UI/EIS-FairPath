# EIS-FairPath

**Fairness-Aware Path-Based Explanations for Dynamic Heterogeneous Graphs via Edge Importance Supervision**

EIS-FairPath explains positive collaboration predictions on dynamic heterogeneous bibliographic graphs. It uses four path-independent gradient probes to estimate physical-event importance, then optimizes a fresh explanation mask with an equal-role KL objective and a path objective. Internal and cross-community events define the structural fairness roles.

This repository contains the current **KL_H4 / MEAN_PATH_V1** implementation (four-probe default, 28 October 2026), three complete DBLP domain graphs, their shared raw source archive, and the calibrated prediction models needed to run explanations.

## Installation

Use Python 3.10 or newer in a virtual environment. Install Git LFS before cloning this repository. The shared raw DBLP archive is stored with Git LFS; retrieve its contents after cloning:

```bash
git lfs install --local
git lfs pull
python -m pip install -e ".[dev]"
python -m eis_fairpath verify-data
python -m pytest -q
```

The implementation runs on CPU. Required numerical libraries are PyTorch, NumPy, NetworkX, and SciPy. SciPy supplies the historical PageRank computation. PyTorch Geometric is unnecessary for this implementation.

The migration was verified with Python 3.13.15, PyTorch 2.13.0+cu132, NumPy 2.5.1, NetworkX 3.5, SciPy 1.16.3, and pytest 8.4.2, using CPU execution and one PyTorch thread per command. The version ranges in `pyproject.toml` describe installation requirements; this list records the environment actually tested.

## Run an explanation

Each command below generates a fresh example pair from the complete domain graph and computes its explanation with the bundled calibrated teacher:

```bash
python -m eis_fairpath explain --dataset mining --output outputs/mining.json
python -m eis_fairpath explain --dataset nlp --output outputs/nlp.json
python -m eis_fairpath explain --dataset vision --output outputs/vision.json
```

The default prediction years are 2008, 2004, and 2007, respectively. The model uses exactly the five preceding calendar years. Explanations require a positive teacher prediction (probability greater than 0.5), complete coverage of those five years, and a nonempty sampled event domain containing both Internal and Cross roles. These are the current method's input conditions; invalid queries return an error.

List example author pairs or supply a new pair and year:

```bash
python -m eis_fairpath queries --dataset nlp --year 2005 --limit 5
python -m eis_fairpath explain --dataset nlp --year 2005 --author-a AUTHOR_ID_A --author-b AUTHOR_ID_B --seed 42 --output outputs/custom.json
```

Replace the author placeholders with IDs returned by `queries`; original author names are also accepted when their normalized identifiers occur in the domain. An explicit `--query-id` provides a stable identity for the importance probes. Reuse that identity across explanation mask seeds to reuse the same probe sequence.

`--root /path/to/EIS-FairPath` selects the repository containing `data/` and `checkpoints/`. After the editable installation, commands also work from another working directory. `--checkpoint` accepts another checkpoint in the portable format documented in [checkpoints/README.md](checkpoints/README.md).

Output JSON includes the physical-event catalog (typed endpoints, external IDs, relations, and years), selected paths, event scores, raw mask values, probe evidence, per-step losses and gradients, and the teacher/support hashes. Event IDs in a path resolve through the `events` mapping. A query with no admissible H4 path returns `output_type: "no_path"`.

## Python interface

```python
from eis_fairpath.checkpoints import load_teacher
from eis_fairpath.graph import load_graph
from eis_fairpath.explainer import prepare_query, explain

graph = load_graph("data/processed/cs_dblp_nlp.json")
teacher = load_teacher("checkpoints/cs_dblp_nlp.pt", dataset=graph.dataset)
query = prepare_query(
    graph, teacher.model,
    ("AUTHOR_ID_A", "AUTHOR_ID_B"), year=2005,
    qid="my-stable-query-id", sampler_config=teacher.sampler,
)
result = explain(query, seed=42)
```

`explain` freezes the teacher, initializes fresh physical-event mask logits, and optimizes only those logits. Its optional `evidence` argument reuses evidence from a previous result with the same query identity, teacher, support, and ordered event/role domain.

## Method defaults

| Component | Setting |
| --- | --- |
| Prediction model | Four-layer structure-aware causal HGT, hidden dimension 128 |
| Fixed support | 1,024 physical events; depth quotas 128/256/320/320; fanout 32 |
| Sampling | Seed 42, native fixed quotas, preserved identity, five calendar years |
| Importance evidence | Four query-seeded antithetic random gate probes |
| Event importance | Mean absolute positive-class logit gradient over the four probes |
| Prediction objective | Mean of Internal/Cross KL(target importance distribution \| mask distribution) |
| Mask distribution | Softmax of raw mask logits within each role, temperature 1 |
| Path objective | Native occurrence-averaged on-path/off-path objective and margin 0.1 |
| Optimization | Fresh Gaussian mask, fresh Adam, learning rate 0.2, 40 updates |
| Search | Exact temporal top-k search using current mask; H4; 41 searches |
| Output | Up to three paths, at most four events each; event budget 15 |

The two role KL terms have equal weight regardless of role size. Constant importance evidence in either role disables the entire KL term through a differentiable zero. The native path objective remains active. Teacher retention and complement forwards are diagnostics; prediction BCE is absent from mask optimization.

Four-hop paths and the three-path limit imply at most 12 distinct selected events, within the declared budget of 15. The 1,024-event sampling budget bounds each prediction's computation support; the complete underlying datasets remain intact.

## Complete datasets

| CLI name | Dataset identifier | Years | Papers | Authors | Physical events |
| --- | --- | --- | ---: | ---: | ---: |
| `mining` | `cs_dblp_mining_ir` | 2003–2010 | 6,865 | 9,959 | 54,701 |
| `nlp` | `cs_dblp_nlp` | 1991–2013 | 8,957 | 8,466 | 58,685 |
| `vision` | `cs_dblp_vision` | 2002–2009 | 9,503 | 11,459 | 66,578 |

`data/processed/` retains every valid publication matching each domain's venue/year selector. The included raw archive and three processed files total 105,159,920 bytes. Hashes and selectors are recorded in `data/manifest.json` and each dataset's `source_manifest`. [DATA_SOURCES.md](DATA_SOURCES.md) describes provenance and the dataset scope.

Regenerate the three domain files into a separate directory and verify their exact hashes:

```bash
python -m eis_fairpath rebuild-data --output rebuilt-data
```

Node features, typed relations, physical-event IDs, annual histories, and structural communities are reconstructed from these complete graph inputs. The public interface accepts new queries directly; it needs no precomputed query cohort.

## Repository layout

```text
eis_fairpath/
  explainer.py       query preparation and complete per-query optimization
  evidence.py        four-probe event importance
  losses.py          equal-role KL and native path loss
  native/           temporal path search and community scoring
  sampling/         causal fixed-support sampling
  models/           prediction model and physical-event gating
  graph.py          complete heterogeneous graph construction
  datasets.py       exact reconstruction from the raw archive
  checkpoints.py    portable frozen-teacher loading
  cli.py            command line interface
data/               full raw and processed DBLP data and hashes
checkpoints/        three calibrated teacher states and hashes
tests/              numerical, graph/model, checkpoint, and CLI checks
```

Migration validation included exact reconstruction of all three processed files, tensor equality for all calibrated checkpoint states, complete explanations on each real domain, and source-versus-extracted numerical comparison. On the same small graph, checkpoint, evidence, and mask seed, all 40 steps' losses, gradients, masks, and paths matched the source kernel exactly; the four probe gate vectors, gradients, and aggregated importance also matched exactly.
