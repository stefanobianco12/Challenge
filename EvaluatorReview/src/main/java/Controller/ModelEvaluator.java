package Controller;

import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;
import bean.Prediction;
import bean.Review;
import com.google.gson.Gson;


public class ModelEvaluator {
    private static final String TEST_DATA_FILE = "src/main/resources/test_set.json";;
    private static final String MODEL_API_URL = "http://localhost:5000/predict";
    private static final HttpClient client = HttpClient.newHttpClient();
    private static final Gson gson = new Gson();
    private static float accFood;
    private static float accDelivery;
    private static float accApproval;


    private static Prediction predictionModel(String text) throws IOException, InterruptedException {
        String requestBody = gson.toJson(text);
        System.out.println(requestBody);
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(MODEL_API_URL))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        return gson.fromJson(response.body(), Prediction.class);
    }

    public static void main(String[] args) throws IOException, InterruptedException {

        List<Review> reviews = gson.fromJson(new FileReader(TEST_DATA_FILE), new com.google.gson.reflect.TypeToken<List<Review>>() {}.getType());
        int cont_food=0;
        int cont_delivery=0;
        int cont_approval=0;
        for (Review review : reviews){
            Prediction prediction=predictionModel(review.getReview());
            if(prediction.getFoodRating()==review.getFoodRating())
                cont_food++;
            if(prediction.getDeliveryRating()==review.getDeliveryRating())
                cont_delivery++;
            if(prediction.getApproval()==review.getApproval())
                cont_approval++;
            System.out.println(prediction.getFoodRating());
        }
        accFood=cont_food/reviews.size();
        accDelivery=cont_delivery/reviews.size();
        accApproval=cont_approval/reviews.size();
        System.out.println("Accuracy food ratings: "+accFood);
        System.out.println("Accuracy delivery ratings: "+accDelivery);
        System.out.println("Accuracy approval: "+accApproval);
    }

}
