# numan : Numerosity analysis
Numan is a Python library for a project aiming to identify the neural correlates of numerosity
abilities in zebrafish. The broad project aims to test the hypothesis that the ability to represent numerosity has an evolutionarily conserved neural basis and to identify the cellular and molecular processes involved. In particular, using a 2P Light-Sheet microscope, we recorded the whole-brain GCaMP activity in zebrafish larvae in response to a change in the number of visual stimuli and we aim to find a set of neurons/activity patterns that is characteristic of a specific number presented to the fish.

Numan contains only the analysis tools and relies on [vodex](https://github.com/LemonJust/vodex) for data management.

<p align="center">
  <img src="img/cover.JPG" alt="cover" width="600"/>
</p>
Schematic of the analysis pipeline used for numerosity-stimuli experiments. The image shows an experiment with two conditions: visual stimuli "2" and "5".

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install numan.

```bash
pip install numan
```

## Usage

Please see notebooks/examples on [github](https://github.com/LemonJust/numan).

## Contributing
Pull requests are welcome, but for now it's only me working on this project, so it might take me some time. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
