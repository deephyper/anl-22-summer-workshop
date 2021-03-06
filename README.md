# anl-22-summer-workshop
Official Repository of the ANL 2022 Summer Workshop

## Local Installation for Developers

1. Install Deephyper by following [documentation of installation instruction](https://deephyper.readthedocs.io/en/latest/install/local.html).

2. Install matplotlib

```console
pip install matplotlib
```

3. Clone the repository

```console
git clone https://github.com/deephyper/anl-22-summer-workshop.git
cd anl-22-summer-workshop/
```

### Notebooks Tests

To test the notebooks directly on Colab with the `develop` branch from deephyper add this in the before the first cells

```ipython
!rm -rf deephyper_repo/
!git clone -b develop https://github.com/deephyper/deephyper.git deephyper_repo
!pip install -e "deephyper_repo/[nas,popt,autodeuq]" --use-feature=in-tree-build
!pip install matplotlib

!git clone https://github.com/deephyper/anl-22-summer-workshop.git
!cd anl-22-summer-workshop/notebooks/
```