##  Mini 2 Lab 3 csagbo
> I implemented a dashboard to visualize the Iris dataset.
> The dashboard is a web application that allows users to interact with the data.
> The dashboard is built using the Dash library in Python. 
> The dashboard has the following features:
> * Explore Iris training data
>   - Load the Iris dataset
>   - Upload a new dataset
>   - Visualize the dataset with different plots (Histogram, Scatterplot)
>   - Dataset overview in a table
> * Build model and perform training. Here we can 
>   - Build a new model by select a given uploaded dataset through it's ID
>   - Retrain an existing model using its ID and a given dataset ID.
> * Score model: 
	- We basically score a row of data which is a set of entries for which we classify the species of Iris. 
> * Test Iris data
> 	- We can test the model with a given dataset ID and get the accuracy of the model.

### Files

> This project files are described below:
> - __*app_lab3_template.py*__ : The main file for the dashboard built with Dash
> - __*app_iris.py*__ : The Rest API written in python with Flask
> - __*base_iris_lab1.py*__ : The base class for the Iris dataset and model
> - __*iris_extended_encoded.csv*__ : The Iris dataset
> - __*request_client.py*__ : The client driver
> - __*logs*__ : Folder containing the logs listed below

### Running the code:
You can run the code by running the following command in the terminal:

- First Terminal to start the Rest API
   - `python3 app_iris.py`
- Second Terminal to start the Dash application
   - `python3 app_lab3_template.py`
   - Make sure to have the  `request_client.py` file in this same directory to test the application.

- When the applications start take the url of the Dash application and paste it in the browser to view the dashboard.

!!! Thank you for reviewing my submission. !!!

