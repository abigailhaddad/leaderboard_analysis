
# Readme for OpenAI Leaderboard Analysis

This project is a Python script that analyzes the performance of models on the OpenAI leaderboard. It does this by scraping commit dates from GitHub for each model and plots their scores over time. The leaderboard can be found at this URL: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

## Inputs

The main input for this script is an Excel file named `raw_leaderboard.xlsx`. This file should contain a sheet named `Sheet1` that has a column called `Model`, which includes the names of the models on the leaderboard, and a column that contains hyperlinks to the GitHub page for each of these models. 

The Excel file is manually created by copying the leaderboard data from the above URL. This data includes the model name and the corresponding performance metrics on four different challenges: ARC, HellaSwag, MMLU, and TruthfulQA.

## Outputs

The output is a PNG image file named `leaderboard_scores.png`. This image file contains a plot of the models' scores over time. It includes plots for the four different challenges and compares the scores of the models with the scores of GPT-3.5 and GPT-4. 

## Procedure

1. The script first reads the Excel file and extracts the hyperlinks from the `Model` column. 

2. Then it uses the `requests` and `BeautifulSoup` libraries to scrape the text from the commit history pages on GitHub for each model.

3. After scraping the text, it extracts the dates of the commits using regular expressions.

4. It standardizes the dates and then uses them to restructure the DataFrame to include only the highest available scores for each challenge on each date.

5. Finally, it plots the scores over time, saves the plot as a PNG image file, and displays it.

Please note that the first commit on the model page is used as the benchmark for the model's availability. This may not perfectly correspond to when the model became available on the leaderboard.

## Future Expansion

Two potential ways to expand this project are:

1. Using Selenium to automatically pull the leaderboard table from the OpenAI site instead of manually copying and pasting it into an Excel file.

2. Utilizing Selenium again to pull date information for the models that require interaction (clicking) on the model page before the commit histories become available. (The models currently getting dropped are: 'llama-65b' 'llama-30b' 'bigcode/starcoderplus' 'llama-13b' 'alpaca-13b'
 'llama-7b' 'alessandropalla/instruct_gpt2')

Please be aware that these extensions would introduce more complexity to the script and would require additional dependencies.

## Dependencies

This project requires the following Python libraries:

* pandas
* openpyxl
* requests
* BeautifulSoup
* re
* datetime
* dateparser
* matplotlib

You can install these libraries using pip:
```
pip install pandas openpyxl requests beautifulsoup4 regex datetime dateparser matplotlib
```

## Execution

To execute the script, navigate to the directory containing the script and the input Excel file, and run the script using a Python interpreter. 

```
python leaderboard_analysis.py
```

The script will output the `leaderboard_scores.png` image file in the same directory.

For any issues, please refer to the original code and accompanying blog post.
