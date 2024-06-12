package bean;

public class Review {
    private String review;
    private int foodRating;
    private int deliveryRating;
    private int approval;

    public Review(String review, int foodRating, int deliveryRating, int approval){
        this.review=review;
        this.foodRating=foodRating;
        this.deliveryRating=deliveryRating;
        this.approval=approval;
    }

    public String getReview() {
        return review;
    }

    public int getDeliveryRating() {
        return deliveryRating;
    }

    public int getFoodRating() {
        return foodRating;
    }

    public int getApproval() {
        return approval;
    }

    public void setFoodRating(int foodRating) {
        this.foodRating = foodRating;
    }

    public void setReview(String review) {
        this.review = review;
    }

    public void setDeliveryRating(int deliveryRating) {
        this.deliveryRating = deliveryRating;
    }

    public void setApproval(int approval) {
        this.approval = approval;
    }
}
