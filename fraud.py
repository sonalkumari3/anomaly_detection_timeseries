### @author: sonal.kumari1910@gmail.com

### result information can be found in a directory named as result inside the present working directory

### importing Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import math

class findFraudUsers:

   def write_fraud_user_data(self, fraud_users, group_approach):
        """
         This function extracts the list of unique fraud/abnormal user id from the given fraud data and write them to the disc in the form of pandas dataframe.
         Parameters
         ----------
             fraud_users:      fraud user data
             threshold:  z_score cut point to differentiate fraud users from normal users
         Returns
         -------
             user_list:  list of unique fraud/abnormal user id
         """
        ## check if dataframe is empty
        if(len(fraud_users)==0):
            print("Fraud data is empty for approach "+ group_approach)
            return []

         ##check the data-type of "fraud_users" and process accordingly
        if(isinstance(fraud_users, list) == False):
            user_list = np.unique(list(fraud_users['user_id']))
        else:
            user_list = np.unique(fraud_users)

        file.write("Total fraud user id using " + group_approach + " : " + str(len(user_list)) + '\n')
        print("Total fraud user id using " + group_approach + " : " + str(len(user_list)))

        if (len(user_list) > 1):
            fraud_user_df = pd.DataFrame(user_list, columns=["user_id"])
            fraud_user_df = fraud_user_df.sort_values(by=['user_id'])
            fraud_user_df.to_csv(base_output + 'fraud_users_' + group_approach + '.csv', index=False)

        return user_list

   def detect_fraud_modified_zscore(self, data, feature, threshold=3.5):
        """
         This function extract list of fraud users based on modified z-score.
         source: https://lavastorm.zendesk.com/hc/en-us/community/posts/360009526673-Outlier-Detection-Using-Modifed-Z-Score
         Parameters
         ----------
             data:                user data
             group_approach:      the name of grouping method: identifier
             threshold:           z_score cut point to differentiate fraud users from normal users: reccomended value should be 3
         Returns
         -------
             user_list:           list of fraud/abnormal user id
         """
        ## make a copy before modifying it
        data_z = data.copy()
        ##feature contains the total search count for each of the user
        data_z['mod_z_Score'] = 0.6745 * (data[feature] - data[feature].median()) / np.median(np.abs(data[feature] - data[feature].median()))
        # max_z= math.ceil(data_z['mod_z_Score'].max())

        ## histogram plot of modified z-score: to identify cut-off point (for abnormal search pattern) by looking at insights
        self.hist_plot(data_z, 'mod_z_Score', "hist_serach_frequency_mod_z_Score.png", 100)

        fraud_users = data_z[data_z['mod_z_Score'] > threshold]
        user_list = self.write_fraud_user_data(fraud_users, 'mod_z_Score')


        return user_list

   def detect_fraud_zscore(self, data, feature, threshold=3):
        """
         This function extract list of fraud users based on z-score.
         Parameters
         ----------
             data:                user data
             group_approach:      the name of grouping method: identifier
             threshold:           z_score cut point to differentiate fraud users from normal users: reccomended value should be 3
         Returns
         -------
             user_list:           list of fraud/abnormal user id
         """
        data_z = data.copy()
        ##feature contains the total search count for each of the user
        data_z['z_Score'] = (data[feature] - data[feature].mean()) / data[feature].std()
        max_z= math.ceil(data_z['z_Score'].max())

        ## histogram plot of z-score: to identify cut-off point (for abnormal search pattern) by looking at insights
        self.hist_plot(data_z, 'z_Score', "hist_serach_frequency_z_Score.png", max_z)

        fraud_users = data_z[data_z['z_Score'] > threshold]
        user_list = self.write_fraud_user_data(fraud_users, 'z_score')


        return user_list

   def find_frauds_ensemble_approach(self, listofList):
        """
         This function extract the common users from the list of fraud users lists.
         Parameters
         ----------
             listofList:    list of list of fraud users
         """
        i=0
        for cl in listofList:
            if(i==0):
                ### initialize with the first list for the first iteration
                common_users=cl
            else:
                ## find the common user ids from the current list and previous iteration computed common user ids
                common_users = list(set(common_users) & set(cl))
            i +=1

        self.write_fraud_user_data(common_users, 'ensemble_approach')

   def fraud_iqr(self, data, feature):
        """
         This function extract list of fraud users based on inter-quartile range (IQR).
         Parameters
         ----------
             data:               user data
             feature:            feature name on which IQR based fraud detection is applied
         Returns
         -------
             user_list:          list of fraud/abnormal user id
         """
        ## compute 1st and 3rd quantile of user search frequency
        q1, q3 = np.percentile(data[feature], [25, 75])
        ### find the difference between 3rd and 1st quantile
        iqr = q3 - q1

        ## extract lower and upper bound based on 1st and 3rd quantile
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        ##extract user data which lies outside the lower and upper bound: abnormal/fraud users
        fraud_user_data = data.loc[((data[feature] < lower_bound) | (data[feature] > upper_bound))]  ##find outlier

        ###write the result
        fraud_user_list = self.write_fraud_user_data(fraud_user_data, 'iqr')

        return fraud_user_list

   def hist_plot(self, df_group, feature, plotname="hist_plot.png", bin_count=36):
        """
         This function plot histogram of user search frequency and save the plot with the given file name.
         Parameters
         ----------
             df_group:      user data frame with frequency information in feature column
             feature:       feature name for which histogram is plotted
             plotname:      file name to save the generated histogram plot
             bin_count:     bin count for histogram plot
         """
        ## plot the given data frequency count (i.e. feature column)
        plt.hist(df_group[feature], bins=list(range(bin_count)))
        #plt.hist(df_group[feature], bins=range(0,150))

        # Add title and axis names
        plt.title('Histogram plot for user search frequency')
        plt.xlabel(feature)
        plt.ylabel('Total number of users')

        ## save the plot to the disc
        plt.savefig(base_output + plotname)
        plt.close()

   def identify_fraud_users_histogram_Approach(self, df_group, threshold, feature):
      """
      This function extract list of fraud users based on search frequency threshold computed through histogram plot.
      Parameters
      ----------
        df_group:           grouped user data with search frequency
        threshold:          a cut point for differentiating fraud users from normal users
        feature:            feature name which is analyzed for finding frauds/anomaly
      Returns
      -------
        user_list:          list of unique fraud/abnormal user id
      """
      ####identify fraud users based on histogram insights
      ## histogram plot of user groups: to identify abnormal search pattern
      self.hist_plot(df_group, feature, "hist_group_search_data.png",df_group[feature].max())

      fraud_users_df = df_group[df_group[feature] >= threshold]
      ###write the result
      fraud_users_ids = self.write_fraud_user_data(fraud_users_df, 'hist_insights')

      return fraud_users_ids

   def fraud_users_identification_techniques(self, df):
      """
      This function applies 4 major techniques to extract list of fraud users: 1) histogram insights based, 2) z-score based, 3) IQR based, and 4) ensemble of all
      """

      ### list of list initialization: to store list of fraud users extracted from various approaches
      LoL = []

      ####identify fraud users based on histogram insights from cumulative user group
      fraud_users_list_group_hist= self.identify_fraud_users_histogram_Approach(df, 50, "freq")

      LoL.append(fraud_users_list_group_hist)

      ### identify fraud users based on z-score

      fraud_users_zscore = self.detect_fraud_zscore(df, "freq", 2.1)
      LoL.append(fraud_users_zscore)

      ### identify fraud users based on modified z-score
      fraud_users_mod_zscore = self.detect_fraud_modified_zscore(df, "freq", 3.5)

      LoL.append(fraud_users_mod_zscore)

      ### identify fraud users based on IQR
      fraud_users_iqr = self.fraud_iqr(df, "freq")
      LoL.append(fraud_users_iqr)

      ###find fraud users by ensembling above list
      findFraudUsersObj.find_frauds_ensemble_approach(LoL)

      ## Flattening the list of list of fraud users
      all_fraud_user_list = list(itertools.chain.from_iterable(LoL))
      ### remove duplicates from the combined fraud user lists (coming from all the techniques applied)
      self.write_fraud_user_data(all_fraud_user_list, 'combined_all_techniques_frauds')


class fileRead:

   def __init__(self, base):
      """
      This function load the data files in the memory.
      Parameters
      ----------
        base:      path where data files are present
      """
      self.signup_df = self.load_data(base + 'signup_data.csv')
      self.signup_df.name = 'Signup'
      self.print_data_stats(self.signup_df)
      self.call_df = self.load_data(base + 'call_data.csv')
      self.call_df.name = 'Call'
      self.print_data_stats(self.call_df)
      self.message_df = self.load_data(base + 'message_data.csv')
      self.message_df.name = 'Message'
      self.print_data_stats(self.message_df)
      self.search_df = self.load_data(base + 'search_data.csv')
      self.search_df.name = 'Search'
      self.print_data_stats(self.search_df)

    ### defining function for extracting dayofyear from datetime column
   def datetime_info(self, df):
      """
      This function extract useful features from datetime column.
      Parameters
      ----------
          df      user data with datetime column
      Returns
      -------
          timeseries_df: dataframe with updated datetime information
      """

      ## make a copy
      timeseries_df = df.copy()
      ### rename *_ts column name to datetime
      timeseries_df.columns = timeseries_df.columns.str.replace('.*_ts', 'datetime')

      ### converting millisecond to datetime
      timeseries_df['datetime'] = pd.to_datetime(timeseries_df['datetime'], unit='ms')
      ###extract dayofyear from datetime column
      timeseries_df['dayofyear'] = pd.to_datetime(timeseries_df['datetime']).dt.dayofyear

      return timeseries_df

   def load_data(self, filename):
      """
      This function load the data in the form of pandas dataframe.
      Parameters
      ----------
          filename:      file name in which data is stored
      Returns
      -------
          df:             loaded input data in the form of pandas dataframe
      """
      df = pd.read_csv(filename, sep='\t')
      df = self.datetime_info(df)
      df = df.sort_values('dayofyear')
      df = df.set_index('dayofyear')

      return df

   def print_data_stats(self, df):
      """
      This function write data information in the file.
      Parameters
      ----------
          df:      data
      """
      file.write("\nPrinting stats for " + str(df.name)+" dataframe : " + '\n')
      file.write("Dataframe shape : " + str(df.shape) + '\n')
      file.write("List of columns : " + str(list(df.columns)) + '\n')
      ## validate total number of unique users in each datasets
      uniue_users_df = df['user_id'].unique()
      file.write("Total number of unique user : " + str(uniue_users_df.shape[0]) + '\n')

class preProcess:

    def remove_valid_users(self, call_df, message_df, search_df):


        ## remove redundant users
        min_date = search_df.index.min()
        max_date = search_df.index.max()
        call_df = call_df.loc[min_date:max_date]
        message_df = message_df.loc[min_date:max_date]
        #print(call_df.head(n=4))
        # call_df = call_df[call_df.index >= min_date]
        # message_df = message_df[message_df.index >= min_date]

        ### get frequency count per user
        grouping_columns = ['user_id']
        search_df_freq = search_df.groupby(grouping_columns).count().reset_index()
        search_df_freq.rename(columns={"datetime": "freq"}, inplace=True)

        call_df_freq = call_df.groupby(grouping_columns).count().reset_index()
        call_df_freq.rename(columns={"datetime": "freq"}, inplace=True)

        message_df_freq = message_df.groupby(grouping_columns).count().reset_index()
        message_df_freq.rename(columns={"datetime": "freq"}, inplace=True)

        # search_df_freq = search_df['user_id'].value_counts().reset_index()



        ## combining call and message user data: valid users
        merged_users_freq = pd.merge(call_df_freq, message_df_freq, on='user_id', how='outer')
        #print(merged_users_freq.head(n=4))
        merged_users_freq = pd.merge(merged_users_freq, search_df_freq, on='user_id', how='outer')

        merged_users_freq = merged_users_freq.fillna(0)


        # ### removing data from valid_users_df of dates exceeding date in search_df
        match_data_lessor = merged_users_freq.loc[merged_users_freq['freq_x'] + merged_users_freq['freq_y'] < merged_users_freq['freq']]
        match_data_lessor.drop(columns=['freq_x', 'freq_y'], inplace=True)

        print("Total unique users who lookup the database without any call/message history: " + str(match_data_lessor.shape[0]))
        file.write(
            "Total unique users who lookup the database without any call/message history: " + str(match_data_lessor.shape[0]) + '\n')
        return match_data_lessor
if __name__ == '__main__':

    # reading all tab seperated data files
    base = './'

    ### creating a text file for writing the result output
    dir_name = 'result/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    base_output = dir_name
    file = open(base_output + 'fraud_detection_result_output.txt', 'w')

    file.write("############ Reading data ###########" + '\n')
    readData = fileRead(base)
    signup_df = readData.signup_df
    call_df = readData.call_df
    message_df = readData.message_df
    search_df = readData.search_df

    file.write("############ Input data loaded in the memory ###########" + '\n')

    file.write('\n\n' + "############ Removing valid users from search data ###########" + '\n')
    preProcessObi=preProcess()
    merged_df = preProcessObi.remove_valid_users(call_df, message_df, search_df)


    file.write('\n\n' + "############ Finding Fraud users from merged_df ###########" + '\n')
    ##merged_df is subset of search data which contains all fraud users
    findFraudUsersObj = findFraudUsers()
    findFraudUsersObj.fraud_users_identification_techniques(merged_df[:])


    file.close()
