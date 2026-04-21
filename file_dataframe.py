

# %%


df_peaks = pd.read_csv( source_dir / "Master_Peak_Log.csv.gz", parse_dates=['timestamp'])
df_dropouts = pd.read_csv( source_dir / "Dropout_Map.csv", parse_dates=['start', 'end'])

# %%

source_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_4\terminal\2512058__SN_921536130')
file_name='HR_Analysis_10s_from_1049.csv'
df_HR_10s = pd.read_csv(source_dir / file_name , parse_dates=['Bin_Start', 'Bin_End'])

dict_attrs = {'path':'batch_4\terminal\2512058__SN_921536130',
              'rec_start_windows': pd.Timestamp('2025-12-05 10:48:33'),
              'gassing_start_windows': pd.Timestamp('2025-12-05 10:49:27')}

df_HR_10s.attrs = dict_attrs

file_name='HR_Analysis_10s_from_1049.pkl'
df_HR_10s.to_pickle( source_dir / file_name  )

# %%% check the time format

df_HR_10s[:4]
    # Out[8]: 
    #             Bin_Start             Bin_End  Average_Heart_Rate_BPM
    # 0 2025-12-05 10:49:27 2025-12-05 10:49:37                   204.0
    # 1 2025-12-05 10:49:37 2025-12-05 10:49:47                   318.0
    # 2 2025-12-05 10:49:47 2025-12-05 10:49:57                   282.0
    # 3 2025-12-05 10:49:57 2025-12-05 10:50:07                   318.0

df_HR_10s['Bin_Start'][4]
    # Out[10]: Timestamp('2025-12-05 10:50:07')

type(df_HR_10s['Bin_Start'][4])
    Out[11]: pandas.Timestamp

# %%

source_name = "df_pulse_binned_5s.pkl"
df_pulse_binned_5s = pd.read_pickle( source_dir / source_name  )

# %%

df_pulse_binned_5s[:5]
    # Out[15]: 
    #                        pressure  pressure_pct_of_baseline  Minutes_from_Gassing
    # 2025-12-05 10:48:30  138.133109                104.638295             -0.950000
    # 2025-12-05 10:48:35  131.820760                 99.856578             -0.866667
    # 2025-12-05 10:48:40  133.655785                101.246643             -0.783333
    # 2025-12-05 10:48:45  136.755675                103.594864             -0.700000
    # 2025-12-05 10:48:50  139.746592                105.860537             -0.616667

df_pulse_binned_5s.attrs
    # Out[16]: 
    # {'rec_start_windows': Timestamp('2025-12-05 10:48:33'),
    #  'gassing_start_windows': Timestamp('2025-12-05 10:49:27')}

df_pulse_binned_5s.attrs = dict_attrs

file_name= 'df_pulse_binned_5s.pkl'
df_pulse_binned_5s.to_pickle( source_dir / file_name  )

# %%

