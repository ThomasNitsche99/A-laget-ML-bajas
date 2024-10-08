### Maskin lærings prosjektet 

#### A-Laget 

* Mina 
* Eivind
* Thomas


### Some notes

#### Prediction Requirements
* For each row in the test dataset, predict the latitude and longitude of the specified **vesselID** at the given **Time** .

#### Time range and Data alignment
* The timestamps in the test dataset start from **20204-05-08**, which is immediately after the last date in the training data.
* This means you need to predict vessel positions starting from one day after the training data ends, extending up to five days into the future.

#### Scaling factor
* The **scaling_factor** is provided for evaluation purposes and indicates the importance of each prediction in the overall metric.
* While you dont need to use it direclty in the model, understanding it can help prioritize accuracy for predictions with heigher weights.

#### Data preporation and Modeling
* Historical data usage: Ensure you have sufficient historical AIS data for each **vesselID** to make predictions more accurate.
* Temporal Modeling: Since predictions are time-dependant, consider using time-series forecasting models or sequence models like LSTMs or GRUs.

#### Modenling approach
* Multi-Output Regression: Since we have two target varibales, long_pred and lat_pred, consider models that can handle multi-output targets.
* Time-Series Models: Use models suited for sequential data. 
