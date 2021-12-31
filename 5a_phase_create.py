"""
    - phase association problem / earthquake detection
      based on systematic travel time moveout across an
      array of n stations

    1) generate random, independent Arrival Time records at n stations

    2) select a random earthquake location within xmin, xmax, ymin, ymax

    3) compute distance to all stations

    4) compute Travel time to all stations using homogeneous velocity structure

    5) Add random origin time to computed travel times


"""
import numpy as np

import matplotlib.pyplot as plt

#=================================0==============================
#                       params and files
#================================================================
iSeed    = 1234
file_out = f"data/arrival_times_train_ran_{iSeed}.txt"

nEq      = 100 # picks with systematic move out
nSta     = 6
nRanAT   = 100 # wrong picks
# location params
xmin_sta, xmax_sta = 0, 10
ymin_sta, ymax_sta = -100, 100 # in km
xmin, xmax = 0, 10
ymin, ymax = 0, 10 # in km
vp         = 6 # km/s
OT_max     = 15 # in seconds
# OT_max factor for false picks
ran_AT_max_fac = 1.5
test_plot  = False

np.random.seed( iSeed)
#=================================1==============================
#                  suite of random arrival times
#================================================================
max_t_ran = np.sqrt( (xmax-xmin)**2 + ( ymax-ymin)**2)+OT_max
m_AT_ran = np.random.uniform( 1, ran_AT_max_fac*max_t_ran, nRanAT*nSta).reshape( nRanAT, nSta)
# add labels
m_AT_ran = np.concatenate(( m_AT_ran, np.zeros((nRanAT,1))), axis = 1)
print( 'shape of random data matrix', m_AT_ran.shape)
#=================================2==============================
#   earthquake and stations within xmin, xmax, ymin, ymax
#================================================================
# determine station locations
a_x_sta = np.random.uniform( xmin_sta, xmax_sta, nSta)
a_y_sta = np.random.uniform( ymin_sta, ymax_sta, nSta)

# determine earthquake locations
a_x_eq  = np.random.uniform( xmin, xmax, nEq)
a_y_eq  = np.random.uniform( ymin, ymax, nEq)

## test plot of station and eq locations
if test_plot == True:
    plt.figure( 1)
    ax = plt.subplot( 111)
    ax.plot( a_x_eq, a_y_eq, 'ko', ms = 2, label = f"events ({nEq})")
    ax.plot( a_x_sta, a_y_sta, 'rv', ms = 4, label = f"stations({nSta})")
    ax.set_xlabel( 'X (km)')
    ax.set_ylabel( 'Y (km)')
    ax.legend()
    plt.axis( 'equal')
    plt.savefig( "5_Eq_sta_map.png")
    #plt.show()

#=================================3==============================
#            distance, travel and arrival time to all stations
#================================================================

m_AT_eqs = np.zeros( (nEq, nSta))
a_r      = np.zeros( nEq)
a_OT_ran = np.random.uniform( 1, OT_max, nEq)# random origin times
for i in range( nSta):
    # distance
    a_r = np.sqrt( (a_x_eq-a_x_sta[i])**2 + (a_y_eq-a_y_sta[i])**2)
    # travel and arrival time
    m_AT_eqs[:,i] = a_r/vp + a_OT_ran
# add labels
m_AT_eqs = np.concatenate(( m_AT_eqs, np.ones((nEq,1))), axis = 1)
# concatenate random and earthquake ATs
m_AT = np.vstack(( m_AT_eqs, m_AT_ran))
print( 'shape of entire input data: ', m_AT.shape)
#------------test plot arrival times---------------------
if test_plot == True:
    plt.figure(2)
    ax = plt.subplot(111)
    for i in range( nRanAT):
        sel = np.argsort(  m_AT_ran[i])
        ax.plot( a_y_sta, m_AT_ran[i,0:-1], 'o-', lw = .2, ms = 5, alpha = .1, label = 'noise')
    # plot moveouts
    for i in range( nEq):
        sel  = np.argsort( a_y_sta)
        ax.plot( a_y_sta[sel], m_AT_eqs[i,0:-1][sel], 'v-', lw = .5, ms = 2)#, label = 'phase picks')

    ax.set_xlabel( 'Northing (km)')
    ax.set_ylabel( 'Arrival Time (sec)')
    plt.savefig( f"5_moveout_ran_{iSeed}.png")
    plt.show()

# shuffle
m_AT = m_AT[np.random.randint(0, nEq+nRanAT, nEq+nRanAT)]



#=================================4==============================
#                    save as ASCII text file
#================================================================
my_fmt = '%6.2f'*nSta
my_head= '  AT_'*nSta
my_head= my_head + '   iEq'
np.savetxt( file_out, m_AT, fmt=my_fmt+'%4i',
            header = my_head)













