### @author: sonal.kumari1910@gmail.com

### result information can be found in a directory named as result inside the present working directory

### importing Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

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
        if(isinstance(fraud_users, list) == "False"):
            user_list = np.unique(list(fraud_users['user_id']))
        else:
            user_list = np.unique(fraud_users)

        file.write("Total fraud user id using " + group_approach + " : " + str(len(user_list)) + '\n')

        if (len(user_list) > 1):
            fraud_user_df = pd.DataFrame(user_list, columns=["user_id"])
            fraud_user_df = fraud_user_df.sort_values(by=['user_id'])
            fraud_user_df.to_csv(base_output + 'fraud_users_' + group_approach + '.csv', index=False)

        return user_list


   def detect_fraud_zscore(self, data, group_approach, threshold):
        """
         This function extract list of fraud users based on z-score.
         Parameters
         ----------
             data:                user data
             group_approach:      the name of grouping method: identifier
             threshold:           z_score cut point to differentiate fraud users from normal users

         Returns
         -------
             user_list:           list of fraud/abnormal user id
         """
        ##datetime contains the total search count for each of the user
        data['z_Score'] = np.abs(stats.zscore(data['datetime']))
        fraud_users = data[data['z_Score'] > threshold]
        user_list = self.write_fraud_user_data(fraud_users, 'z_score_'+group_approach)

        return user_list

   def find_common_user(self, list1, list2):
        """
         This function extract list of common users from the given two lists.
         Parameters
         ----------
             list1:  a list of user id
             list2:  another list of user id

         Returns
         -------
             common_list:  a list of common user list
         """
        common_list = list(set(list1) & set(list2))
        return common_list

    ### identify fraud users based on iqr
   def fraud_iqr(self, data, group_approach):
        """
         This function extract list of fraud users based on inter-quartile range (IQR).
         Parameters
         ----------
             data:               user data
             group_approach:     the name of grouping method: identifier

         Returns
         -------
             user_list:          list of fraud/abnormal user id
         """
        ## compute 1st and 3rd quantile of user search frequency
        q1, q3 = np.percentile(data['datetime'], [25, 75])
        ### find the difference between 3rd and 1st quantile
        iqr = q3 - q1

        ## extract lower and upper bound based on 1st and 3rd quantile
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        ##extract user data which lies outside the lower and upper bound: abnormal/fraud users
        fraud_user_data = data.loc[((data['datetime'] < lower_bound) | (data['datetime'] > upper_bound))]  ##find outlier

        ###write the result
        fraud_user_list = self.write_fraud_user_data(fraud_user_data, 'iqr_' + group_approach)

        return fraud_user_list

   def hist_plot(self, df_group, plotname="hist_plot.png"):
        """
         This function plot histogram of user search frequency and save the plot with the given file name.
         Parameters
         ----------
             df_group:      user data frame with frequency information in datetime column
             plotname:      file name to save the generated histogram plot

         """
        ## plot the given data frequency count (i.e. datetime column)
        plt.hist(df_group['datetime'], bins=list(range(36)))

        # Add title and axis names
        plt.title('Histogram plot for user search frequency')
        plt.xlabel('Number of search frequency')
        plt.ylabel('Total number of users')

        ## save the plot to the disc
        plt.savefig(base_output + plotname)
        plt.close()

   def identify_fraud_users_histogram_Approach(self, df_group, threshold, group_approach):
      """
      This function extract list of fraud users based on search frequency threshold computed through histogram plot.
      Parameters
      ----------
        df_group:           grouped user data with search frequency
        threshold:          a cut point for differentiating fraud users from normal users
        group_approach:     the name of grouping method: identifier

      Returns
      -------
        user_list:          list of unique fraud/abnormal user id
      """
      ####identify fraud users based on histogram insights
      ## histogram plot of user groups: to identify abnormal search pattern
      self.hist_plot(df_group, "hist_group_search_data"+group_approach+".png")

      fraud_users_df = df_group[df_group['datetime'] >= threshold]
      fraud_users_ids = np.unique(list(fraud_users_df['user_id']))
      file.write("Total fraud user id based on user-group histogram insights : " + str(len(fraud_users_ids)) + '\n')

      if (len(fraud_users_ids) > 1):
         fraud_user_df = pd.DataFrame(fraud_users_ids, columns=["user_id"])
         fraud_user_df = fraud_user_df.sort_values(by=['user_id'])
         fraud_user_df.to_csv(base_output + 'fraud_users_hist_insights_'+group_approach+'.csv', index=False)

      return fraud_users_ids

   def fraud_users_identification_techniques(self):
      """
      This function applies 3 major techniques to extract list of fraud users: 1) histogram insights based, 2) z-score based, 3) IQR based
      """

      all_fraud_user_list = []

      ####identify fraud users based on histogram insights from cumulative user group
      fraud_users_list_group_hist= self.identify_fraud_users_histogram_Approach(df_group, 30, "_based_on_user_id")
      all_fraud_user_list == all_fraud_user_list.extend(fraud_users_list_group_hist)

      ## extract fraud users badsed on histogram insights on daywise groups: to identify abnormal search pattern on daily-basis
      fraud_users_list_group_hist_d = self.identify_fraud_users_histogram_Approach(df_daywise_group, 13, "_based_on_user_id_and_dayofyear")
      all_fraud_user_list == all_fraud_user_list.extend(fraud_users_list_group_hist_d)

      ### identify fraud users based on z-score
      fraud_users_zscore = self.detect_fraud_zscore(df_group, "based_on_user_id", 1)
      fraud_users_zscore_d = self.detect_fraud_zscore(df_daywise_group, "based_on_user_id_and_dayofyear", 1)
      all_fraud_user_list == all_fraud_user_list.extend(fraud_users_zscore)
      all_fraud_user_list == all_fraud_user_list.extend(fraud_users_zscore_d)

      ### identify fraud users based on IQR
      fraud_users_iqr = self.fraud_iqr(df_group, "based_on_user_id")
      fraud_users_iqr_d = self.fraud_iqr(df_daywise_group, "based_on_user_id_and_dayofyear")
      all_fraud_user_list == all_fraud_user_list.extend(fraud_users_iqr)
      all_fraud_user_list == all_fraud_user_list.extend(fraud_users_iqr_d)

      common_user = findFraudUsersObj.find_common_user(fraud_users_zscore_d, fraud_users_list_group_hist_d)
      file.write("Total common fraud users from hist and daywise z-score : " + str(len(common_user)) + '\n')

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
      # timeseries_df.rename(columns={timeseries_df.columns.tolist()[-1]: 'datetime'}, inplace=True)
      # timeseries_df.rename(columns={"signup_ts": "datetime"}, inplace=True)

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


if __name__ == '__main__':

    # reading all tab seperated data files
    base = '../'

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
    ### remove valid users from search data table: extract those serach users who are not in call or sms datasets
    file.write("nitially, Search data size= " + str(search_df.shape[0]) + ' & unique users= ' + str(search_df['user_id'].unique().shape[0])+ '\n')

    df_search_sample = search_df[~search_df['user_id'].isin(call_df['user_id'])]
    file.write("After removing call users, size= " + str(df_search_sample.shape[0]) + ' & unique users= ' + str(df_search_sample['user_id'].unique().shape[0])+'\n')

    df_search_sample = df_search_sample[~df_search_sample['user_id'].isin(message_df['user_id'])]
    file.write("After removing message users, size= " + str(df_search_sample.shape[0]) + ' & unique users= ' + str(df_search_sample['user_id'].unique().shape[0])+'\n')

    file.write('\n\n' + "Creating groups of remaining users in search data based on user_id and day of year" + '\n')
    ## sort extracted serach user data for further insight generation
    # df_search_sample = df_search_sample.sort_values(by=['user_id', 'datetime'], ascending=False)

    ### grouping serach users based on user_id
    df_group = df_search_sample.groupby(['user_id']).count().reset_index()
    file.write("Total number of Groups of tentative user data shape based on user_id : " + str(df_group.shape[0]) + '\n')

    ### grouping serach users based on user_id and day of year
    df_daywise_group = df_search_sample.groupby(['user_id', 'dayofyear']).count().reset_index()
    file.write("Total number of Groups of tentative user data shape based on user_id & dayofyear : " + str(df_daywise_group.shape[0]) + '\n')

    file.write('\n\n' + "############ Finding Fraud users in search data with tentative fraud users ###########" + '\n')
    findFraudUsersObj = findFraudUsers()
    findFraudUsersObj.fraud_users_identification_techniques()


    file.close()



