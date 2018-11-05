# Hackathon Three: Sentiment Analysis

The competition runs from Friday the 2nd of November until noon on Monday the 5th of November. Submissions will be graded throughout this period. The winning solution will reflect on the autograder leaderboard. We ask the winning team to present their solution to us on Wednesday the 7th of November.

We’ll provide snacks on Friday but lunch is your business.


## The task

You are provided with a dataset of 500,000 reviews. Each review has a user provided rating from 1 to 5. The task is to build a model that can accurately predict the rating given only the text. As this is an NLP task you are permitted to use pretrained language models or word embeddings. You are not however permitted to use other sentiment datasets. The data is provided in the link below:


You should find a zipped folder with two csv files, `train_data.csv` and `train_response.csv`. Each file will have a `review_id` column and text or stars respectively.

## Making submissions

We have created an autograder for this hackathon. Unlike Kaggle we do not grade your submissions nor do we provide a test dataset. We take your preprocessing pipeline and apply it directly to the data then fit your model. Rather you will do this in a script called `run_model.py` which we run with two keyword arguments. The first argument is `--input` which takes the path of a csv data file. The second argument is `--submission` which is the path where your predictions will be written. Bear in mind that your submission should return the `review_id` and columns corresponding to each rating and the associated probability of the text generating that rating. It is important to note that we will not be supporting `R` submissions for this hackathon.

Your `run_model.py` script, Pipfile, Pipfile.lock, pretrained model and any pickled objects you may have should all sit in a zipped folder which you will submit to the autograder. We will assume [pipenv](https://pipenv.readthedocs.io/en/latest/) for dependency management so Pipfile and Pipfile.lock are required. We set a runtime limit of 2 minutes and an upload limit of 1GB.

Your submissions will be graded using [cross entropy loss](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html) which will reflect as your score on the leaderboard. You can find the autograder on the link below. At the moment we only support `bcx.co.za`, `explore-datascience.net` and `explore-ai.net` email addresses.

