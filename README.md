# **Autonomous Systems Interview Practice** 

## Perception/Sensor Engineer

[//]: # (Image References)

[image1]: ./images/undistort.png ""
[image2]: ./images/threshold.png ""
[image3]: ./images/perspective.png ""
[image4]: ./images/histogram.jpg ""
[image5]: ./images/sliding_window.jpg ""
[image6]: ./images/search_prior.jpg ""
[image7]: ./images/sensor_comparison.png ""
[image8]: ./images/sensor_data_comparisons.png ""
[image9]: ./images/async_predict_update.png ""

1. **Required question**
Explain a recent project you've worked on. Why did you choose this project? What difficulties did you run into this project that you did not expect, and how did you solve them?

__Answer__:
Recently I have completed a behavioral cloning project. In this project I have got some training data, The car have to learn the driving behavior from the training data, and clone its behavior. I choose this project is because it is very interesting and useful. With this project, we can teach the car how to drive. I think the most challenge part of this project is that I have to train a model from scratch to teach the vehicle how to drive in the center of the lane. The overall strategy for deriving a model architecture was reference to End-to-End Deep Learning for Self-Driving Cars. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I modified the model and added some dropout and regularizer layer in the original model.
The final step was to run the simulator to see how well the car was driving around track one.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. To augment the data sat, I do some random flip, shift, rotate. In each batch, I randomly generate 60% training data. After the collection process, in each epoch I had almost 400,000 of data points. I then preprocessed this data by crop lane portion of the image and resize to [66, 200] as model requirement. Then convert to YUV color space. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the total lost not decreased anymore. I used an adam optimizer so that manually training the learning rate wasn't necessary.

2. Explain your lane detection algorithm. How can it be improved? How does it account for curves and hills? How could you integrate machine learning techniques into the algorithm?

__Answer__:
1. Lane detection algorithm
    1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    2. Apply a distortion correction to raw images.
    3. Use color transforms, gradients, etc., to create a thresholded binary image.
    4. Apply a perspective transform to rectify binary image ("birds-eye view").
    5. Detect lane pixels and fit to find the lane boundary.
    6. Determine the curvature of the lane and vehicle position with respect to center.
    7. Warp the detected lane boundaries back onto the original image.
    8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

2. Need improve
    1. Changed lane

        Because the polynomial calculation is depended on previous experience. If user change lane, the lane detection algorithm would be fail. So, if user changes lane, I need to recalculate lane polynomial.
    
    2. Abnormally detected Lane

        If the lane have some lane like shadow or cavity, it may affect my lane detection. I need to improve lane detection algorithm.
    
    3. Curvature too large.

        If the curvature is too large, my lane detection algorithm may fail to detect lane with the background. I may need to increase the degree of polynomial.

3. Account for curves and hills

    1. Apply Histogram

        get histogram of the thresholded perspective transformed image.

    2. Sliding Window Search

        calculate 2nd order polynomial.

    3. Search from Prior

        recalculate polynomial at time t+1, according to time t.

4. integrate machine learning

    With the lane detection algorithm, we can correctly identify lanes. Then we can start to collect training data. If it exceeds the lane boundary we can give it a negative reward, else gives a positive reward. And apply reinforcement learning to let the vehicle learn how to drive into the lane boundary.

![alt text][image1]

![alt text][image2]

![alt text][image3]

| Apply Histogram     | Sliding Window Search | Search from Prior   |
|---------------------|-----------------------|---------------------|
| ![alt text][image4] | ![alt text][image5]   | ![alt text][image6] |

3. What are some of the advantages & disadvantages of cameras, lidar and radar? What combination of these (and other sensors) would you use to ensure appropriate and accurate perception of the environment?

__Answer__:
The comparison of camera, laser and radar is as below. 

![alt text][image7]

From this, we can find that camera is good at resolution, noise and size. Radar is good at velocity, weather and size. And lidar is ok at resolution.

From the functionality point of view, we can refer below figure.

![alt text][image8]

We can find that camera is good at object classification, lane tracking. Lidar is good at object detection and poor lighting. Radar is good at velocity, bad weather and poor lighting. We can have some combination to get the most benefit of camera, lidar and radar.

4. [Code] What approach would you take if the various sensors you are using have different refresh rates?

__Answer__:
We can let each sensor maintain its own prediction update scheme. The prediction step of all sensors should be the same. But for the update step, because each sensor has its own measurement unit. We need to asynchronous update measurement for all sensors. Take radar, lidar sensor as example. Because the coordinate system of radar and lidar are different. Radar use Polar coordinate system, lidar use Cartesian. For each radar measurement, we first need to convert from Cartesian to Polar coordinate system. Then do further measurement update.

![alt text][image9]

```c++
void KalmanFilter::Predict() {
  /**
   * TODO: predict the state
   */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * TODO: update the state by using Kalman Filter equations
   */
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */
// Convert from Cartesian to Polar, START[
  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);

  double rho = sqrt(px*px + py*py);
  double theta = atan2(py, px);
  double rho_dot = (px*vx + py*vy) / rho;
  VectorXd h = VectorXd(3);
  h << rho, theta, rho_dot;
// Convert from Cartesian to Polar, END]
  VectorXd y = z - h;
  while ( y(1) > M_PI || y(1) < -M_PI ) {
    if ( y(1) > M_PI ) {
      y(1) -= M_PI;
    } else {
      y(1) += M_PI;
    }
  }

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;
  // New state
  x_ = x_ + (K * y);
  int x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
```