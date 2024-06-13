# Challenge
The application java controller, given a test set json file, evaluates the model. It iteratively makes a HTTP POST request to the server, where is instanced the model. It sends each textual review to the server where exsposed a RESTful API POST to predict the values: food rating, delivery rating, approval and an approval's reliability estimation, sending to the application.<br>
Periodically a scheduler job retrain the model given a train and validation set if the accuracy of each class prediction is above or equal a certain threshold parameter, the job save the new model and trigger the server with a HTTP POST to update it. 

## Model description and design
I interpret the output prediction as categorical variable so the model is a multi-classification task problem based on the sentiment analysis of the reviews. <br>
To extract the input features I use a pretrained transformer from hugginface: BERT and subsequently the tokenizer for the input data. Typically it is used for downstream task. <br>
For each class, a one layer shallow neural networks. One layer only for simplicity, 2 layers is enoug with a ReLu actiovation function on the hidden layer. <br>
The model archicture is the following <br>
----        BERT Transformer          ---- <br>
-- 1-layer NN   1-layer NN   1-layer NN -- <br>
<br>
In the forward phase, it outputs only the logits values of each network because since in the training phase the loss function to optimize is the sum of two cross-entropy loss and one BCEloss function, each loss need in input the logit value. <br>
<br>
To convert in probability I used a softmax function for the food and delivery rating and a sigmoid function for the approval since is a binary value. The approval estimation reliabiliy is computed as abs(sigmoid(approval_logit)-0.5). More the sigmoid values is near 1 or 0 more the prediction is reliability.<br>

## Training dataset generation
Several strategies to build a training dataset:<br>
--Web scraping from the food delivery sites using BeatifulSoup to parse HTML and XML and Selenium   for scraping dynamical data.<br>
<br>
-- or a simple one to build a mock dataset is to generate synthetic data making prompt engineering on a LLM such as GPT3 or LLama.<br>
For each strategies is important to build a balanced dataset, in particular the data equally distributed for each task class value.<br>
<br>
For simplicity I followed the second strategies, where i made a query to generate a review for food and delivery could be a food and deliery rating X Y.

## Evaluation metric
For simplicy, I consider the accuracy metric individually for each task: food rating, delivery rating and approval.<br>
Other metric can be computed such as precision, recall, F1.

## Deployment & tools
In my setup the model prediction function is exposed by a RESTful API service using the Flask library. Since the service has to interarct with different distributed modules the best option is to deploy with REST, it is a lightweight protocol, flexible and easy to use it.<br>
<br>
In a real scenario, the service, the scheduler and the evaluation model application  are containerized using Docker. This ensures that they  can run in any environment that supports Docker, making it easier to integrate with various applications.<br>
After, the containereized applications are deployed on a Kubernetes cluster, where manages the load balacing. The REST API service is exsposed on a kubernetes node to communicate to the outside the cluster.<br>
<br>
The Java application evaluator iteratively sends one review at time, since the test set can be large it is a strong limitation. One possible solution is to use the network file system of Kubernets, for sharing the dataset across pods or nodes. So the evaluator controller triggers only the evaluation test set phase.


## Run
As first, run the API service on a terminal:
<pre>
pip install -r ./model/requirements.txt
py ./model/controller
</pre>
run the job scheduler on another terminal:
<pre>
py ./model/scheduler
</pre>
For the EvaluatorReview application, it runs on an Java IDE, possibily on Intellij. If necessary, export the Jar library: com.google.code.gson


