import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, fftpack, stats, signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Module to plot curves.
def myPlot(dy):
    dx = [j * 5 for j in range(0, len(dy))]
    plt.plot(dx, dy)
    plt.show()

# Module to plot PCA.
def plotPca():
    attributes = ['component_1', 'component_2', 'component_3', 'component_4', 'component_5']
    number_attributes = len(attributes)
    for i in range(len(pca_components)):
        value = list(pca_components.iloc[i])    
        value += value[:1]
        angles = [n / float(number_attributes) * 2 * np.pi for n in range(number_attributes)]
        angles += angles[:1]
        ax = plt.subplot(111, polar=True)
        #Add the attribute labels to our axes
        plt.xticks(angles[:-1],attributes)
        #Plot the line around the outside of the filled area, using the angles and values calculated before
        ax.plot(angles,value)
        #Fill in the area plotted in the last line
        ax.fill(angles, value, 'teal', alpha=0.1)
        labl = 'Timeseries_' + str(i)
        #Give the plot a title and show it
        ax.set_title(labl)
        plt.show()

# Module to interpolate values.
def interpolate_glucose_vals(i):
    s1 = []
    s2 = []
    store_g_val = []
    store_t_val = []
    x = 0
    y = 0
    for elems in glucose_df.iloc[i]:
        s1.append(elems)
    for elems in datenum_df.iloc[i]:
        s2.append(elems)
    data = {'times' : s2, 'vals' : s1}
    df = pd.DataFrame(data = data)
    # Interpolate the date num values.
    df['times'].interpolate(inplace = True)
    store_t_val = list(df['times'])
    df.set_index("times", inplace = True)
    # Interpolate the glucose values.
    df['new_vals'] = df['vals'].interpolate(method = 'polynomial', order = 5)
    store_g_val = list(df['new_vals'])
    for cols in glucose_df.columns:
        glucose_df.at[i, cols] = store_g_val[x]
        x += 1
    for col in datenum_df.columns:
        datenum_df.at[i, col] = store_t_val[y]
        y += 1

# Module to perform polynomial fit
# to get the coefficient values.
def perform_polyfit(i):
    colmns = ['coeff_0', 'coeff_1', 'coeff_2', 'coeff_3']
    co_eff = []
    itr = 0
    g_lvl = list(glucose_df.iloc[i])
    interval = [j * 5 for j in range(0, len(glucose_df.iloc[i]))]
    p_fit = list(np.polyfit(interval, g_lvl, 3))
    co_eff.extend(p_fit)
    # Plot chart
    # Uncomment the below lines to
    # plot the curve.
    plt.plot(p_fit)
    plt.show()
    for cols in colmns:
        glucose_features.at[i, cols] = co_eff[itr]
        itr += 1

# Module to perform fft.
def performFft(i):
    itr = 0
    g_lvl = list(glucose_df.iloc[i])
    fft_plot = abs(fftpack.fft(g_lvl))
    fft_vals = sorted(set(fft_plot), reverse = True)
    glucose_features.at[i, 'high_1'] = fft_vals[1]
    glucose_features.at[i, 'high_2'] = fft_vals[2]
    glucose_features.at[i, 'high_3'] = fft_vals[3]
    # Plot chart
    # Uncomment the below lines to
    # plot the curve.
    #print('FFT', [fft_vals[1], fft_vals[2], fft_vals[2]])
    plt.plot(fft_vals[1:])
    plt.show()

# Module to perform CGm velocity method.
def cgmVelocity(i):
    window_size = 3
    time_line = 15
    velocity = []
    val_store = glucose_df.iloc[i]
    for j in range(0, len(glucose_df.iloc[i]) - window_size):
        interim = (val_store[j] - val_store[j + window_size]) / time_line
        velocity.append(interim)
    # Find standard deviation of the series.
    s_dev = pd.Series(velocity).std()
    mean_val = pd.Series(velocity).mean()
    median_val = pd.Series(velocity).median()
    glucose_features.at[i, 'cgm_velocity_stdv'] = s_dev
    glucose_features.at[i, 'cgm_velocity_mean'] = mean_val
    glucose_features.at[i, 'cgm_velocity_median'] = median_val
    plt.plot(velocity)
    plt.show()

# Module to perform Welch method.
def performWelch(i):
    store_interim = glucose_df.iloc[i]
    hz, welch_values  = np.array((signal.welch(store_interim)))
    welch_std = pd.Series(welch_values).std()
    welch_mean = pd.Series(welch_values).mean()
    welch_median = pd.Series(welch_values).median()
    glucose_features.at[i, 'max_welch'] = max(welch_values)
    glucose_features.at[i, 'std_welch'] = welch_std
    glucose_features.at[i, 'mean_welch'] = welch_mean
    glucose_features.at[i, 'median_welch'] = welch_median
    plt.plot(hz, welch_values)
    plt.show()

# Module to perform PCA.
def performPCA():
    features = glucose_feature_matrix.columns
    feature_matrix = glucose_feature_matrix.loc[:, features].values
    # Normalize the feature values.
    feature_matrix = stats.zscore(feature_matrix)
    pca_cons = PCA(n_components = 5)
    principal_components = pca_cons.fit_transform(feature_matrix)
    final_component = pd.DataFrame(data = principal_components, 
                                   columns = ['component_1', 'component_2', 
                                              'component_3', 'component_4', 
                                              'component_5'])
    pca_var = pca_cons.explained_variance_ratio_
    pc_comps = (abs(pca_cons.components_))
    #print(abs(pca_cons.components_))
    pca_var = ['{:f}'.format(item) for item in pca_var]
    #print(pca_var)
    return final_component, pc_comps, pca_var

# Module to know the top features in PCA.
def findTopfeatuesinComponents():
    colmns = glucose_feature_matrix.columns
    for i in range(0, 5):
        component_name = 'PC_'+ str(i + 1)
        interim = sorted(just_comps[i], reverse = True)
        fst_max = interim[0]
        scnd_max = interim[1]
        print(component_name)
        print('------------')
        print('Top 1st Feature', colmns[list(just_comps[i]).index(fst_max)])
        print('Top 2nd Feature', colmns[list(just_comps[i]).index(scnd_max)])

# Driver Module
if __name__ == '__main__':
    datenum_df = pd.DataFrame(columns = ['cgmDatenum_ 1', 'cgmDatenum_ 2', 'cgmDatenum_ 3', 'cgmDatenum_ 4',
       'cgmDatenum_ 5', 'cgmDatenum_ 6', 'cgmDatenum_ 7', 'cgmDatenum_ 8',
       'cgmDatenum_ 9', 'cgmDatenum_10', 'cgmDatenum_11', 'cgmDatenum_12',
       'cgmDatenum_13', 'cgmDatenum_14', 'cgmDatenum_15', 'cgmDatenum_16',
       'cgmDatenum_17', 'cgmDatenum_18', 'cgmDatenum_19', 'cgmDatenum_20',
       'cgmDatenum_21', 'cgmDatenum_22', 'cgmDatenum_23', 'cgmDatenum_24',
       'cgmDatenum_25', 'cgmDatenum_26', 'cgmDatenum_27', 'cgmDatenum_28',
       'cgmDatenum_29', 'cgmDatenum_30', 'cgmDatenum_31'])
    
    glucose_df = pd.DataFrame(columns = ['cgmSeries_ 1', 'cgmSeries_ 2', 'cgmSeries_ 3', 'cgmSeries_ 4',
       'cgmSeries_ 5', 'cgmSeries_ 6', 'cgmSeries_ 7', 'cgmSeries_ 8',
       'cgmSeries_ 9', 'cgmSeries_10', 'cgmSeries_11', 'cgmSeries_12',
       'cgmSeries_13', 'cgmSeries_14', 'cgmSeries_15', 'cgmSeries_16',
       'cgmSeries_17', 'cgmSeries_18', 'cgmSeries_19', 'cgmSeries_20',
       'cgmSeries_21', 'cgmSeries_22', 'cgmSeries_23', 'cgmSeries_24',
       'cgmSeries_25', 'cgmSeries_26', 'cgmSeries_27', 'cgmSeries_28',
       'cgmSeries_29', 'cgmSeries_30', 'cgmSeries_31'])
    
    for i in range(1, 6):
        datenum_file_path = 'Data/DataFolder/CGMDatenumLunchPat' + str(i) + '.csv'
        glucose_file_path = 'Data/DataFolder/CGMSeriesLunchPat' + str(i) + '.csv'
        datenum_interim_df = pd.read_csv(datenum_file_path)
        glucose_interim_df = pd.read_csv(glucose_file_path)
        datenum_df = datenum_df.append(datenum_interim_df.iloc[:, 0 : 31], ignore_index = True, sort = False)
        glucose_df = glucose_df.append(glucose_interim_df.iloc[:, 0 : 31], ignore_index = True, sort = False)
    
    # Interpolate the missing values.
    for i in range(len(glucose_df)):
        interpolate_glucose_vals(i)
    
    # Store the index of rows with NA.
    nan_rows = pd.isnull(glucose_df).any(1).to_numpy().nonzero()[0].tolist()
    # Drop the rows with NA value.
    glucose_df = glucose_df.dropna()
    datenum_df = datenum_df.drop(nan_rows)
    #plt.plot(datenum_df.iloc[0], glucose_df.iloc[0])
    #plt.show()
    # Reset the indexes after the
    # deop.
    glucose_df.reset_index(drop = True, inplace = True)
    # Convert all the data into integer type.
    glucose_df = glucose_df.astype(np.float64)
    # Use this new data frame to
    # calculate new features.
    glucose_features = glucose_df.copy()
    print('Tasks')
    print('-----')
    print('Q A. The different features-')
    print('Polyfit')
    print('FFT')
    print('CGM Velocity')
    print('Welch')
    
    print('-----')
    print('Q B. The charts in features-')
    '''
        Description: 1. Polyfit feature handler.
        @author: Suryadeep
        @Created: 7th Feb
    '''
    # Add the feature columns to
    # dataframe.
    glucose_features['coeff_0'] = -1
    glucose_features['coeff_1'] = -1
    glucose_features['coeff_2'] = -1
    glucose_features['coeff_3'] = -1
    # Call the poly fit feature.
    for idx in range(0, len(glucose_features)):
        if idx not in nan_rows:
            perform_polyfit(idx)
    '''
        Description: 2. FFT feature handler.
        @author: Suryadeep
        @Created: 7th Feb
    '''
    # Add the feature column to
    # dataframe.
    # Call the perfrom fft feature.
    for idx in range(0, len(glucose_features)):
        performFft(idx)
    '''
        Description: 3. CGM Velocity
        @author: Suryadeep
        @Created: 7th Feb
    '''
    for idx in range(0, len(glucose_features)):
        cgmVelocity(idx)
    '''
        Description: 4. Welch Method
        @author: Suryadeep
        @Created: 7th Feb
    '''
    for idx in range(0, len(glucose_features)):
        performWelch(idx)
    # Final feture matrix to use during PCA.
    glucose_feature_matrix = glucose_features.loc[:, 'coeff_0':'median_welch'].copy()
    print('-----')
    print('Q D. Feature Matrix-')
    print(glucose_feature_matrix)
    
    print('-----')
    print('Q E. PCA-')
    # Get the final feature dataframe
    # after performing PCA.
    pca_components, just_comps, varnces = performPCA()
    print('Variances')
    print(varnces)
    print(pca_components)
    # Plot the PCA spyder charts.
    print('Spyder Charts')
    plotPca()
    # Find the top features
    print('-----')
    print('Q F. Top Features')
    findTopfeatuesinComponents()