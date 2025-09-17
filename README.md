# Automata Object Detection

This repository contains the source code for the Automata Object Detection project. The project is part of the European Project Automata and aims to detect irregular archaeological pieces of ceramics or lithics.

## Description

The Automata Object Detection project is a deep learning-based system that uses computer vision techniques to detect and classify irregular archaeological pieces of ceramics or lithics. The system is designed to assist in the analysis and identification of these artifacts, which can be challenging due to their complex and varied shapes.

## Installation

To install the project, follow these steps:

1. Clone the repository: `git clone --recurse-submodules https://github.com/GDqbrobotics/automata_object_detection-.git`
2. Navigate to the project directory: `cd automata_object_detection-`
3. Build the Docker image: `docker compose build`

## Usage

To use the project, follow these steps:

1. Connect a realsense camera d415 to your computer.
2. Run the object detection script inside the Docker container: `docker compose up` (you don't need to build if  you just change the object_extraction.py file)
The script will process each frame from the camera and save the results to a result.png.

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue. To contribute to the codebase, follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes and commit them: `git commit -m "Your commit message"`
4. Push your changes to your forked repository: `git push origin feature/your-feature`
5. Submit a pull request

## License

This project is licensed under the [...] License.

## Acknowledgements

The Automata Object Detection project is part of the European Project Automata.

## Contact

If you have any questions or need further assistance, please contact giuliano.dami@qbrobotics.com.
