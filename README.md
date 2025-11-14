# Demand Forecasting Via Machine Learning
This is a portion of my capstone submission for my bachelor's degree in computer science. Long before I had embarked on a computer science degree, I thought the world of demand planning and demand forecasting was ripe for machine learning. Organizations often have mountains of clean, historic item demand data that's perfect for 
machine learning. There can also be a wide range of inputs that impact that demand, like seasonality, weather, holidays, etc. The capstone was a ML project so it felt like a good opportunity to put the idea to the test. 
It was my first pass at ML so I learned a lot about the process and particularly about the linear regression model used here.
<br><br>
This was written in Visual Studio Code, with the [main.py](https://github.com/josh-mccurry/ML_Demand_Forecasting/blob/main/main.py) orchestrating instantiation of the different working classes and serving as an outline for when I ultimately would transfer it to Google Colab. 

## What is Demand Forecasting?
Any organization that sells goods at scale grapples with the question of how much of a particular good they need. Unsold stock represents capital tied up in product and is also a major liability. Unsold product can expire or become damaged and never get turned into revenue. 
For this reason, organizations perform demand planning and forecasting. This is the process of looking at historic sales data and making predictions based on that data. This then informs purchasing habits.

## The Data

For my first outing I selected what I considered the bare minimum features from those available to make for a clear example. Below is an example of the data you can find in the [demand_data.csv](https://github.com/josh-mccurry/ML_Demand_Forecasting/blob/main/demand_data.csv) file. 

+ **Warehouse** - The facility the demand was allocated to or serviced from  
+ **Item** - The internal organizational item number for the product sold  
+ **Demand Date** - The date for which the demand was captured  
+ **Original Demand Quantity** - The actual demand quantity that was captured. From this you might infer that for there to be an "original" value, that may mean there is an adjusted value and you would be correct.
+ This gives the inventory management teams discrete control over the demand represented and provides a "what-if" dial in the data.  

| Warehouse | Item | Demand Date | Original Demand Quantity |
| --- | --- | --- | --- |
| 1 | 7830864 | 9/10/25 | 40 | 

## Data Prep
Much of the data related activities are contained within the [data_handler.py](https://github.com/josh-mccurry/ML_Demand_Forecasting/blob/main/data_handler.py) class. The function names and comments should sufficiently describe the activities, but broadly speaking it either preparing the data itself for training or 
managing the results for display in the various visualizations I experimented with.  

## Model Selection
The project called for the selection of one machine learning model, so I landed on the linear regression model for this particular application. The linear regression model effectively finds a "best fit" slope for the trend line across the demand data. For this reason, it has a number of strengths as a proof of concept. 
The first is it would convey the general trend for the demand. Sales teams always want to see measurable, incrementing growth and this would facilitate displaying the overall trend for a given item in a way that could easily be missed in the reams of item demand data. The second is that because it's establishing a best fit 
trend line, it can make predictions that exceed the values it saw in the training data. In other words if the highest demand it ever saw in the data for a widget was quantity of 150, it could predict 151 or more despite that never being represented in the data. This is in contrast to other models such as Random Forest that 
are constrained by the highest values it was exposed to in the training set. Finally, the linear nature would highlight where it doesn't work at all. Items with wild seasonality or highly variable demand will stand out and highlight where it is not the ideal model.

## Testing

The overall methodology of this project began with a top-down software architecture stage, where I determined the work that needed to be done, broke it down into logical groups that could form classes, and built the framework. This ensured I could account for all the task requirements and include them in my broader plan.

From there, the development was performed in a waterfall fashion, where I started at the beginning. I used [main.py](https://github.com/josh-mccurry/ML_Demand_Forecasting/blob/main/main.py) as the general timeline to test the individual classes and methods as they were developed. This testing-in-place and general arrangement worked out exceptionally well.

Finally, I replaced my [main.py](https://github.com/josh-mccurry/ML_Demand_Forecasting/blob/main/main.py) file with the Google Colab Notebook and used the main.py file to inform the general flow of the notebook. The format of the notebook allowed me to easily instantiate those classes linearly, confirm I could get a result, then move on to the next step all while experimenting with issues in my main.py file without impacting the notebook.

The final testing stage was done many times as I made small improvements to the classes, then confirmed the behavior in the notebook. This was repeated many times to ensure the notebook was providing the expected outcomes.

## Accuracy Assessment

I selected the Mean Average Percentage Error score as the measure of how well the model performed. This score performs the following:

1.	Obtains the absolute value of the difference between the Actual Dependent Variable value and the Predicted Dependent Variable value.
2.	Divides each difference obtained in step 1 by the Actual Dependent Variable value to obtain a percentage.
3.	Obtains the average by summing the percentage obtained in step 2, then dividing by the total count of differences calculated.

This was selected for two reasons. The first is that the Mean Average Percentage Score could be implemented against a single line of demand prediction and provide a meaningful score of how well that prediction performed. The second is that it would still provide the overall score of how the model performed, in addition to the prior. 

For this reason, the accuracy of the model can be assessed in two dimensions. The first is the overall actual MAPE score. This score, scaled appropriately, is 400% and I think that is an objective failure. If this were as far as I went, I think I’d feel the entire project was a total failure. 
However, when we look at the individual item performance scores in the visualization, we get a more nuanced look at that score.

<picture>
 <img alt="An image of a bar chart representing the MAPE scores by item by year." src="https://github.com/josh-mccurry/Demand-Forecasting-via-Machine-Learning/blob/main/assets/mape_scores.jpg">
</picture>

This visualization shows a much more optimistic picture. There are very clearly some items in the range that are drastically impacting the MAPE value. Moreover, there are some very high-performing items. We can also see that for some of these poor-performing items, it’s older data that is doing the most damage. 
Conversely, for many (not all) of the performant items, they don’t have as much data available.

This suggests to me that training models by item and experimenting with the frame of time used to train that item-specific model could yield very effective results in the final predictions.

## Lessons Learned
I gleaned a great deal from this project and definitely walked away with some lessons learned. The first is that where I trained the model against all of the items, I think I'd have had more success training a model per item and aggregating their individual predictions. This would make grading a particular model 
against a specific item a simple task. At some point I may add a scaler method and an additional model or two in the interest of tackling the more volatile items. This also makes me consider a more enterprise or production environment where I would want it to intelligently parse my item data and just select the correct model
without requiring user input. That case might require a feature that expresses the relative volatility of that item that could then be used to fit the item to the appropriate model. Overall I was heartened by those items with a lower MAPE score than I was discouraged by the standouts that performed poorly and I think sufficient time,
planning, and development could produce a demand forecasting module entirely leveraging machine learning that produced good results.
