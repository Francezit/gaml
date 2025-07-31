# GAML - Genetic Algorithm with Machine Learning

The project implements various optimization algorithms to recreate reference images using colored polygons.

If you use GAML in your research, please cite the following paper:

Machine Learning and Genetic Algorithms: A case study on image reconstruction
http://dx.doi.org/10.1016/j.knosys.2023.111194



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
