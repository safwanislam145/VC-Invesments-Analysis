# Company Similarity Checker

Welcome to the Company Similarity Checker! This project includes data preprocessing, analysis, and visualization tools to calculate and display the similarity between companies based on various attributes.

## Contents

### Files
- `Preprocessing.ipynb`: Jupyter Notebook for cleaning and preprocessing raw data (`data.csv`).
- `grouped_analysis.ipynb`: Jupyter Notebook for performing grouped data analysis on the cleaned dataset.
- `cleaned_data.csv`: Preprocessed data generated from the `Preprocessing.ipynb`.
- `web_app.py`: Python script for a web application to visualize the cleaned and analyzed data.

### Datasets
- `data.csv`: The raw dataset used for preprocessing.
- `cleaned_data.csv`: The processed dataset ready for analysis and visualization.

## Setup Instructions

### Prerequisites
Ensure the following are installed on your system:
- Python (version 3.8 or higher)
- Jupyter Notebook or Jupyter Lab

### Required Python Packages
- Pandas
- NumPy
- streamlit
- Matplotlib
- Seaborn
- Scikit-learn (if applicable)
- Any other packages listed in the notebooks or scripts

### Install Python Packages
Run the following command to install the necessary packages:
```
pip install pandas numpy streamlit matplotlib seaborn scikit-learn
```

## Instructions to Run the Project

1. Preprocess the Data
    - Run the `Preprocessing.ipynb` notebook to preprocess the raw data. This will generate a `cleaned_data.csv` file.
    - Steps:
      - Open the `Preprocessing.ipynb` notebook in Jupyter.
      - Follow the instructions in the notebook to execute each cell.
      - The output file (`cleaned_data.csv`) will be saved in the project directory.

2. Perform Grouped Analysis
    - Run the `grouped_analysis.ipynb` notebook to analyze the cleaned dataset.
    - Steps:
      - Open the `grouped_analysis.ipynb` notebook in Jupyter.
      - Ensure `cleaned_data.csv` is in the same directory.
      - Execute the cells to generate analysis insights.

3. Run the Web Application
    - Use the `web_app.py` script to launch a web application that visualizes the data.
    - Steps:
      - Open a terminal or command prompt.
      - Run the following command:
         ```
         streamlit run web_app.py
         ```

## Project Workflow

1. Data Input: Start with `data.csv`.
2. Preprocessing: Use `Preprocessing.ipynb` to clean and transform data.
3. Analysis: Explore grouped statistics and insights with `grouped_analysis.ipynb`.
4. Visualization: Launch `web_app.py` for an interactive experience.

## File Descriptions

- `Preprocessing.ipynb`: Cleans and prepares the raw data for analysis.
- `grouped_analysis.ipynb`: Performs detailed analysis and visualizations on the cleaned data.
- `web_app.py`: Implements a web application to visualize the processed and analyzed data.
- `data.csv`: Raw data to be cleaned and analyzed.
- `cleaned_data.csv`: Output of the preprocessing step.

## Notes

- Modify file paths in the scripts if running in a different directory.
- Update the `web_app.py` to customize routes or visualizations as needed.
- For large datasets, consider optimizing the preprocessing steps in `Preprocessing.ipynb`.

## Support

For any questions or issues, feel free to reach out!