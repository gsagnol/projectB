north_pole = (90,0)
weight_limit = 1000#TMP 1000
sleigh_weight = 10


import pandas as pd
import numpy as np
from haversine import haversine
import sub_mlp as subMLT

res = []
for iter in range(500):
    #----------------------#
    #construct an instance #
    #----------------------#
    gifts = pd.read_csv('../input/gifts.csv')
    ngifts = gifts.count()[0]
    centerid = np.random.randint(ngifts)
    center = gifts.loc[centerid][['Latitude','Longitude']].values
    dist_to_center = lambda x: haversine(tuple(x),center)
    accept_proba = 0.25 + np.random.rand() * 0.72

    dists = np.apply_along_axis(dist_to_center,1,gifts[['Latitude','Longitude']].values)
    close_gifts = sorted([(d,gft_id_minus_1) for gft_id_minus_1,d in enumerate(dists)])[:3000]
    wgt = 0.
    sub_gift_ids = []
    for d,gft_id_minus_1 in close_gifts:
        if np.random.rand() > accept_proba:
            continue
        wgt += gifts.loc[gft_id_minus_1].Weight
        if wgt < weight_limit:
            sub_gift_ids.append(gft_id_minus_1+1)
        else:
            break

    #-------------------------#
    #centroid of the instance #
    #-------------------------#
    lats = gifts.loc[[i-1 for i in sub_gift_ids]].Latitude
    longs = gifts.loc[[i-1 for i in sub_gift_ids]].Longitude
    centroid = (lats.mean(),longs.mean())
    dists_to_centroid = [haversine((lats.iloc[i],longs.iloc[i]),centroid) for i in range(len(lats))]
    dist_to_pole = haversine(centroid,north_pole)
    mlt = subMLT.MLT(gifts,sub_gift_ids)
    total_weight = sum(mlt.weights)
    item_weights = mlt.weights[1:]

    wcentroid = (sum(lats.values * item_weights)/sum(item_weights),sum(longs.values * item_weights)/sum(item_weights))
    dists_to_wcentroid = [haversine((lats.iloc[i],longs.iloc[i]),wcentroid) for i in range(len(lats))]
    wdist_to_pole = haversine(wcentroid,north_pole)

    #-----------------------------#
    # Approximate sol of instance #
    #-----------------------------#
    qopt = mlt.quick_opt()

    print iter
    thisres = {'centroid':centroid,
               'wcentroid':wcentroid,
               'total_weight':total_weight,
               'item_weights':item_weights,
               'dists':dists_to_centroid,
               'wdists':dists_to_wcentroid,
               'pole':dist_to_pole,
               'wpole':wdist_to_pole,
               'qopt':qopt.wlatency
               }
    print qopt.wlatency,centroid,sum(dists_to_centroid)

    res.append(thisres)

f = open('analysis_statopt','w')
f.write(str(res))
f.close()

#-----------------#
# Result Analysis #
#-----------------#

f = open('analysis_statopt','r')
res = eval(f.read())
f.close()

vqopt = [r['qopt'] for r in res]
linopt = [r['qopt']-(r['total_weight']+sleigh_weight)*r['pole'] for r in res]
wlinopt = [r['qopt']-(r['total_weight']+sleigh_weight)*r['wpole'] for r in res]

sumdis_to_c = [sum(r['dists']) for r in res]
sumdis_to_wc = [sum(r['wdists']) for r in res]
sumwdis_to_c = [sum([wi*li for wi,li in zip(r['item_weights'],r['dists'])]) for r in res]
sumwdis_to_wc = [sum([wi*li for wi,li in zip(r['item_weights'],r['wdists'])]) for r in res]


figure(1)
scatter(sumdis_to_c,linopt)
figure(2)
scatter(sumwdis_to_c,linopt)
figure(3)
scatter(sumdis_to_wc,wlinopt)
figure(4)
scatter(sumwdis_to_wc,wlinopt)

outliers = [i for i,r in enumerate(res) if sum([wi*li for wi,li in zip(r['item_weights'],r['wdists'])])>278500]
xi =  [sum([wi*li for wi,li in zip(r['item_weights'],r['wdists'])]) for i,r in enumerate(res) if i not in outliers]
yi =  [r['qopt']-(r['total_weight']+sleigh_weight)*r['wpole'] for i,r in enumerate(res) if i not in outliers]
a = np.linalg.lstsq(np.atleast_2d(xi).T,np.atleast_2d(yi).T)[0][0,0]
std(yi/a-xi)

xi =  [sum(r['wdists'])  for i,r in enumerate(res) if i not in outliers]
yi =  [r['qopt']-(r['total_weight']+sleigh_weight)*r['wpole'] for i,r in enumerate(res) if i not in outliers]
np.linalg.lstsq(np.atleast_2d(xi).T,np.atleast_2d(yi).T)

"""
The smallest square errors are attained for the following model: [figure 3]

y = d(wcentroid,pole)*(W + 10) + 75.563 * sum_i d(gift_i,wcentroid)

"""
scatter(vqopt,[(r['total_weight']+sleigh_weight)*r['wpole']+ 75.563*sum(r['wdists']) for r in res])