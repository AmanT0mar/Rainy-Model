# Rainfall prediction model

## Getting Started

### Prerequisites

* [Python 3.11](https://www.python.org/downloads/) or later


### Installation

1. Clone the repo
```sh
git clone https://github.com/AmanT0mar/Rainy-Model.git
```

2. Install python packages
```sh
pip install -r requirement.txt
```
or
```sh
python -m pip install -r requirement.txt
```
or
```sh
python3 -m pip install -r requirement.txt
```

### Usage

1. To start the server, run the following command
```sh
uvicorn main:app --reload
```

## API endpoints

* To get list of cities which are available for prediction
```/city_list``` 

* To get rainfall amount prediction using model
```/rainfall/{city_name}```

## License

[LICENSE](LICENSE.txt)