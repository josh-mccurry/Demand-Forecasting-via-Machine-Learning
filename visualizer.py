import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import warnings


class visualizer:

    warnings.filterwarnings("ignore", message="The palette list has.*")

    # Creates a histogram from the dataframe passed to it
    def better_hist(self, df):
        sb.histplot(df, x="year", y="Original Demand Quantity",
            bins=30, discrete=(True, False)
        )
        plt.xlabel('Years')
        plt.ylabel('Demand Quantity')
        plt.title('Overall Demand')
        plt.show()
        return
    
    # Creates a lineplot of overall demand
    def lineplot(self, df):
        custom_palette = sb.color_palette("husl", 10)
        df.reset_index()
        # Make it a little bigger to give better line separation maybe
        plt.figure(figsize=(9, 7))
        # Create the lineplot
        sb.lineplot(data=df, legend=True, palette=sb.color_palette(custom_palette), dashes=False)
        plt.legend(loc='upper left')
        
        # Labels
        plt.xlabel('Years')
        plt.ylabel('Demand')
        plt.title('Overall Demand')
        plt.show()
        return
    
    def heatmap(self, df):
        # Once I loaded a full file, it was just too short
        #plt.figure(figsize=(10, 50))

        # Create the heatmap
        sb.heatmap(df, annot=False, cmap='crest', vmin=0, vmax=1000)

        # Labels
        plt.xlabel('Year')
        plt.ylabel('Item')
        plt.title('Heatmap of Item Demand Over Time')
        # Had to find these functions to fix the labels
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        plt.show()
        return
    
    def mape(self, results):
        custom_palette = sb.color_palette("husl", 10)
        plt.figure(figsize=(12, 5))
        sb.barplot(results, x='Item', y='MAPE', hue="year", palette=sb.color_palette(custom_palette))

        plt.xlabel('Item')
        plt.ylabel('MAPE Score')
        plt.title('MAPE Scores by Item by Year')
        plt.xticks(rotation=90)
        plt.show()
        return
    

