# Data Reduction Tools


## PypeIt

For general data reduction, [PypeIt](https://pypeit.readthedocs.io/en/release/index.html) is probably a good option.

### Installation

It is recommended to install it on an anaconda environment:

```code
conda create -n pypeit python=3.9
conda activate pypeit
```

and install it using pip:

```code
pip install "pypeit[pyqt5,bottleneck,test]"
```

To install it in developer mode (e.g. if you want to [add a new telescope/instrument](https://pypeit.readthedocs.io/en/release/new_spectrograph.html)), use:

```code
git clone https://github.com/pypeit/PypeIt.git
cd PypeIt
pip install -e ".[dev,pyqt5]"
```

### Test

A simple test can be run with:

```code
run_pypeit -h
```

or for developers:

```code
cd ~/PypeIt
pytest
```

A suite of tests can be found in [PypeIt-development-suite](https://github.com/pypeit/PypeIt-development-suite) for a more in-depth testing.

### Usage

You can follow the [PypeIt Cookbook](https://pypeit.readthedocs.io/en/release/cookbook.html) although it might not be simple to follow.

Below I explain without details a minimal usage.

#### 0. Prepare the data

Assuming you are only using a single intrument, split the data in different directories for different nights.

#### 1. First Execution

First execute `pypeit_setup` like this:

```code
pypeit_setup -r path_to_your_raw_data/ -s <spectrograph>
```

where `<spectrograph>` is the name of the spectrograph (e.g. `ntt_efosc2`).

#### 2. Inspect the outputs

Check the files created in the `setup_files` folder. You will see different configurations separated in different blocks (e.g. `A`, `B`, etc.). If you are happy with the configurations, move to the next step.

#### 3. Second execution: Write the pypeit file for one or more setups 

You can run one (e.g. `A`) or more configurations (e.g. `all`):

```code
pypeit_setup -r path_to_your_raw_data/ -s <spectrograph> -c all
```

#### 4. Run the reduction

The main script to run the PypeIt reduction is `run_pypeit`:

```code
run_pypeit path_to_your_reduction_file/<spectrograph>_>setup>.pypeit -o
```

## Custom Codes

There are also custom codes one can use.

### INT - IDS

I found a code in [this repo](https://github.com/aayush3009/INT-IDS-DataReduction).
