# GAML - Genetic Algorithm Machine Learning Framework

A Python framework for image reconstruction using metaheuristic optimization algorithms. The project implements various optimization algorithms to recreate reference images using colored polygons.

## Overview

GAML (Genetic Algorithm Machine Learning) is a research framework that applies evolutionary computation and other metaheuristic algorithms to solve the image reconstruction problem. The system attempts to recreate target images by evolving a collection of semi-transparent polygons with optimized positions, colors, and transparency values.

## Features

### Implemented Algorithms

- **GA** - Genetic Algorithm with elitism
- **GAML** - Genetic Algorithm with Machine Learning (dynamic parameter adaptation)
- **ILS** - Iterated Local Search
- **TS** - Tabu Search
- **AIS** - Artificial Immune System (Clonal Selection Algorithm)

### Image Quality Metrics

- **MSE** - Mean Squared Error
- **SSIM** - Structural Similarity Index
- **PSNR** - Peak Signal-to-Noise Ratio
- **LOSS** - Custom loss function
- **CP** - Custom performance metric

### Dynamic Parameter Adaptation

The ML variants (GAML, ILSML) feature adaptive parameter control that automatically adjusts algorithm parameters based on search progress using trend analysis.

## Requirements

```bash
pip install Pillow numpy scikit-image opencv-python matplotlib deap sewar
```

## Project Structure

```
gaml/
├── src/
│   ├── main.py                  # Main entry point
│   ├── algorithmBase.py         # Base classes for algorithms
│   ├── GA.py                    # Genetic Algorithm implementation
│   ├── GAML.py                  # GA with Machine Learning
│   ├── ILS.py                   # Iterated Local Search
│   ├── TS.py                    # Tabu Search
│   ├── AIS.py                   # Artificial Immune System
│   ├── imageHelper.py           # Image processing utilities
│   ├── statisticHelper.py       # Statistics and logging
│   └── dynamicParamaters.py     # Dynamic parameter adaptation
├── images/                      # Sample images for testing
└── README.md
```

## Usage

### Basic Usage

```bash
python src/main.py -a GA -d ./images -o ./output
```

### Command Line Arguments

- `-d, --image_folder`: Path to folder containing target images (default: ./images)
- `-a, --algorithm`: Algorithm to use (GA, GAML, ILS, ILSML, TS, AIS)
- `-o, --output_folder`: Output directory for results
- `-v, --verbose`: Enable verbose output
- `-i, --config_file`: Configuration file path
- `-s, --seed`: Random seed value
- `-c, --custom_params`: Custom parameters (space-separated key=value pairs)
- `-t, --irace_output`: Output file for irace integration
- `-n, --irace_id`: Identifier for irace runs

### Configuration File

Create a configuration file to customize algorithm parameters:

```ini
[DEFAULT]
polygon_size = 3
number_of_polygon = 100
max_generation = 1000
population_size = 200
prob_crossover = 0.9
prob_mutation = 0.5
objective_fun_method = MSE
save_image_each = 100
```

### Example Commands

```bash
# Run Genetic Algorithm on all images in ./images folder
python src/main.py -a GA -d ./images -o ./results -v

# Run GAML with custom parameters
python src/main.py -a GAML -d ./images -o ./results -c "max_generation=2000 population_size=300"

# Run with configuration file
python src/main.py -a ILS -d ./images -o ./results -i config.ini -v

# Run with specific seed for reproducibility
python src/main.py -a AIS -d ./images -o ./results -s 42
```

## Algorithm Parameters

### Genetic Algorithm (GA/GAML)

- `population_size`: Population size (default: 200)
- `prob_crossover`: Crossover probability (default: 0.9)
- `prob_mutation`: Mutation probability (default: 0.5)
- `hall_of_fame_size`: Elite individuals preserved (default: 20)
- `crowding_factor`: Diversity control parameter (default: 10.0)

### Iterated Local Search (ILS)

- `pertubation_factor`: Perturbation strength (default: 0.1)
- `neighbor_size`: Neighborhood search size (default: 10/20)
- `hamming_distance`: Mutation distance (default: 1)

### Tabu Search (TS)

- `pertubation_factor`: Perturbation strength (default: 0.1)
- `tabu_list_size`: Size of tabu list (default: 10)

### Artificial Immune System (AIS)

- `number_of_antibodies`: Initial antibody population (default: 100)
- `clone_rate`: Cloning rate (default: 0.1)
- `mutation_exp`: Mutation exponent (default: 0.4)
- `max_antibodies`: Maximum antibodies maintained (default: 100)

### Common Parameters

- `polygon_size`: Number of vertices per polygon (default: 3)
- `number_of_polygon`: Number of polygons in solution (default: 100)
- `max_generation`: Maximum iterations (default: 1000)
- `max_time`: Maximum execution time in seconds (default: -1, disabled)
- `objective_fun_method`: Fitness metric (MSE, SSIM, PSNR, LOSS, CP)
- `save_image_each`: Save intermediate results every N generations (default: 1000)
- `target_solution`: Stop when reaching target fitness (default: -1, disabled)

## Output

For each processed image, the framework generates:

- **results/**: Directory containing intermediate and final images
  - `{generation}_compare.png`: Side-by-side comparison with target
  - `{generation}_generated.bmp`: Generated image
  - `{generation}_solution.txt`: Solution parameters (JSON format)
- **statistic.txt**: Detailed execution statistics
- **inputs.txt**: Algorithm configuration used
- **dynamic_log.txt**: Dynamic parameter changes (ML variants only)

## Research Applications

This framework is designed for:

- Evolutionary art and computational creativity research
- Metaheuristic algorithm comparison studies
- Dynamic parameter adaptation research
- Image approximation and compression studies
- Multi-objective optimization research

## Integration with irace

The framework supports integration with the irace package for automatic algorithm configuration:

```bash
python src/main.py -a GAML -d ./images -o ./results -t irace_output.txt -n run_001
```

## Contributing

Contributions are welcome! To add a new algorithm:

1. Create a new file following the naming convention (e.g., `NEWNAME.py`)
2. Implement `NEWNAMEConfig` inheriting from `AlgorithmConfigBase`
3. Implement `NEWNAME` inheriting from `AlgorithmBase`
4. Implement the `executive()` method with your algorithm logic

## License

This project is available for academic and research purposes. Please cite appropriately if used in publications.


## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed

   ```bash
   pip install Pillow numpy scikit-image opencv-python matplotlib deap sewar
   ```
2. **Memory Issues**: Reduce `population_size` or `number_of_polygon` for large images
3. **Slow Performance**: Consider using smaller images or reducing `max_generation`
4. **No Output Images**: Check that the output directory has write permissions

### Performance Tips

- Use SSIM metric for better perceptual quality
- Start with smaller polygon counts (50-100) for faster convergence
- Use ML variants (GAML, ILSML) for better parameter adaptation
- Enable verbose mode (`-v`) to monitor progress

## Contact

For questions, issues, or contributions, please open an issue in the repository or contact the maintainers.
