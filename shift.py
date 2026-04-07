
# calculates the time shifts.
    # difference of times registered by Notocord relative to Windows.

# %% shift

# this tests the shift in time from notocord relative to Windows.

from datetime import datetime, timedelta

# %%% add

# notocord : bottom of the window.
    # start_time + elapsed_time
    # this returns the current-time that notocord shows ( different from system time ).

def add_time(t1, t2):
    base = datetime.strptime(t1, "%H:%M:%S")
    h, m, s = map(int, t2.split(":"))
    return (base + timedelta(hours=h, minutes=m, seconds=s)).time()


# %%% subtract

# current time differences : 
        # notocord ( A )  -  Windows ( Aw )

def time_diff(t1, t2):
    t1_dt = datetime.strptime(t1, "%H:%M:%S")
    t2_dt = datetime.strptime(t2, "%H:%M:%S")
    return t1_dt - t2_dt


# %% batch-4

# batch-4
# F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\screenshot\2025-12-05

# %%% notocord time

# there should not be any 'space' charachter in the timestamps , otherwise :
    # ValueError: unconverted data remains:

print(add_time("10:37:41", "00:05:57"))
print(add_time("11:10:06", "00:08:02"))
print(add_time("11:45:18", "00:08:00"))

    # 10:43:38
    # 11:18:08
    # 11:53:18

# %%% difference


A  = "10:43:38"
Aw = "09:46:51"

B  = "11:18:08"
Bw = "10:21:25"

C  = "11:53:18"
Cw = "10:56:33"

print(time_diff(A, Aw))
print(time_diff(B, Bw))
print(time_diff(C, Cw))

    # 0:56:47
    # 0:56:43
    # 0:56:45

# so notocord time is about 0:56:43 more than system time.
# these shift were registered in the corresponding table :
    # F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4  |  table_batch_4_.docx

# %% batch-3

# batch-3
# F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\screenshot\batch_3\2025-9-26

# %%% notocord time

print(add_time("09:13:09" , "00:32:56"))
print(add_time("10:21:17" , "00:03:49"))     # file : 2509264
print(add_time("10:21:17" , "00:06:19"))     # file : 2509264

# 09:46:05
# 10:25:06
# 10:27:36

# %%% difference

A  = '09:46:05'
Aw = '09:46:50'

B  = '10:25:06'
Bw = '10:25:51'

C  = '10:27:36'
Cw = '10:28:14'

# windows time is higher : hence : aw - a.
    # otherwise the output will be : -1 day, 23:59:15.
print(time_diff(Aw, A))
print(time_diff(Bw, B))
print(time_diff(Cw, C))

# 0:00:45
# 0:00:45         # file : 2509264
# 0:00:38         # file : 2509264

# these shift were registered in the corresponding table :
    # F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_3  |  print_table_....docx

# %%'







