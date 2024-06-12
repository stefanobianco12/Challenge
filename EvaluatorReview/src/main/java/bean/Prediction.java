package bean;

public class Prediction {
    private int foodRating;
    private int deliveryRating;
    private int approval;
    private float estimateReliability;

    public Prediction(int foodRating,int deliveryRating,int approval, float estimateReliability){
        this.foodRating=foodRating;
        this.deliveryRating=deliveryRating;
        this.approval=approval;
        this.estimateReliability=estimateReliability;
    }

    public int getFoodRating() {
        return foodRating;
    }

    public int getDeliveryRating(){
        return deliveryRating;
    }

    public int getApproval(){
        return approval;
    }

    public float getEstimateReliability(){
        return estimateReliability;
    }
}

