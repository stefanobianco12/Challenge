# Challenge
The application java controller, given a test set json file, evaluates the model. It iteratively makes a HTTP POST request to the server, where is instanced the model. It sends each textual review to the server where exsposed a RESTful API POST to predict the values: food rating, delivery rating, approval and an approval's reliability estimation, sending to the application.
Periodically a scheduler job retrain the model given a train and validation set if the accuracy of each class prediction is above or equal a certain threshold parameter, the job save the new model and trigger the server with a HTTP POST to update it.

## Model description and design
I interpret the output prediction as categorical variable so the model is a multi-classification task problem based on the sentiment analysis of the reviews.
To extract the input features I use a pretrained transformer from hugginface: BERT and subsequently the tokenizer for the input data. Typically it is used for downstream task.
For each class, a one layer shallow neural networks. One layer only for simplicity, 2 layers is enoug with a ReLu actiovation function on the hidden layer.
The model archicture is the following
----        BERT Transformer          ----
-- 1-layer NN   1-layer NN   1-layer NN --

In the forward phase, it outputs only the logits values of each network because since in the training phase the loss function to optimize is the sum of two cross-entropy loss and one BCEloss function, each loss need in input the logit value.

To convert in probability I used a softmax function for the food and delivery rating and a sigmoid function for the approval since is a binary value. The approval estimation reliabiliy is computed as abs(sigmoid(approval_logit)-0.5). More the sigmoid values is near 1 or 0 more the prediction is reliability.


## Training dataset generation
Several strategies to build a training dataset:
--Web scraping from the food delivery sites using BeatifulSoup to parse HTML and XML and Selenium   for scraping dynamical data.

-- or a simple one to build a mock dataset is to generate synthetic data making prompt engineering on a LLM such as GPT3 or LLama.
For each strategies is important to build a balanced dataset, in particular the data equally distributed for each task class value.

For simplicity I followed the second strategies, where i made a query to generate a review for food and delivery could be a food and deliery rating X Y.




