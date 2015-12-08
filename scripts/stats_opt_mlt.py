north_pole = (90,0)
weight_limit = 1000#TMP 1000
sleigh_weight = 10


import pandas as pd
import numpy as np
from haversine import haversine
import sub_mlp as subMLT
import scipy.stats as sst
import scipy.optimize

gifts = pd.read_csv('../input/gifts.csv')
ngifts = gifts.count()[0]

circle = False


def random_weight():
    #TODOPT avec un seul triangle ?
    if np.random.rand()<0.6:
        #unif from -18.5 to 15
        x = sst.uniform(-18.5,15+18.5).rvs()
    else:
        #triangle from 15 to 15+45.98
        x = sst.triang(c=0,loc=15,scale=45.98).rvs()
    return min(max(x,1.),50.)


res = []
for iter in range(500):
    if circle:
        #----------------------#
        #construct an instance #
        #----------------------#
        centerid = np.random.randint(ngifts)
        center = gifts.loc[centerid][['Latitude','Longitude']].values
        dist_to_center = lambda x: haversine(tuple(x),center)
        accept_proba = 0.25 + np.random.rand() * 0.72
        #for proba=1 tends to make disc or radius 160km

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

        positions = None
        width = None
        height = None

    else:
        #----------------------------#
        #construct a random instance #
        #----------------------------#

        center = (45,0.)
        #lognormal or uniform over [0.001,1]
        if np.random.rand() < 0.5:
            ratio_width_height = 10**(-3+np.random.rand()*3)
        else:
            ratio_width_height = max(np.random.rand(),0.001)
        area = 75000+np.random.randint(3)*25000.
        width = 10. + np.random.rand()*140.
        height = (area/ratio_width_height)**0.5
        width = area/height

        lamax = 45.
        dx = lambda x: haversine((45,0.),(x,0.))-height
        lamin = scipy.optimize.fsolve(dx,0)
        dy = lambda y: haversine((45,0.),(45,y))-0.5 * width
        lomin = scipy.optimize.fsolve(dy,-20)
        if lomin>0: lomin = -lomin
        lomax = -lomin

        pos = center
        wgt = 0.

        positions = []
        weights = []

        while True:
            wg = random_weight()
            wgt += wg

            if wgt < weight_limit:
                positions.append(pos)
                weights.append(wg)
                pos = (lamin + np.random.rand()*(lamax-lamin),lomin + np.random.rand()*(lomax-lomin))
                pos = (pos[0][0],pos[1][0])
            else:
                break

        mlt = subMLT.MLT(None,None,weights,positions)
        wcentroid = None
        centroid = None
        total_weight = sum(mlt.weights)
        item_weights = mlt.weights[1:]
        dists_to_wcentroid = 0.
        dist_to_pole = haversine(north_pole,center)
        wdist_to_pole = None
        dists_to_centroid = None
        dists_to_wcentroid = None


    #-----------------------------#
    # Approximate sol of instance #
    #-----------------------------#
    #qopt = mlt.quick_opt()
    qopt = mlt.optimize(disp=1)

    print iter
    thisres = {'centroid':centroid,
               'wcentroid':wcentroid,
               'total_weight':total_weight,
               'item_weights':item_weights,
               'dists':dists_to_centroid,
               'wdists':dists_to_wcentroid,
               'pole':dist_to_pole,
               'wpole':wdist_to_pole,
               'qopt':qopt.wlatency,
               'positions': positions,
               'width':width,
               'height':height,
               'area': area,
               'ratio': ratio_width_height,
               'iter':iter
               }
    print qopt.wlatency,centroid,sum(dists_to_centroid),width,height

    res.append(thisres)

f = open('analysis_statopt_trueopt_3areas','w')
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



#------------------------------------#
# Result Analysis -- thin rectangles #
#------------------------------------#

f = open('analysis_statopt_rectangles','r')
res = eval(f.read())
f.close()

const_1 = [ (r['total_weight']+sleigh_weight)*r['pole'] for r in res ]
const_2 = [ (r['total_weight']+sleigh_weight)*(r['pole']+0.5*r['height']) for r in res ]
const_3 = [ sum([haversine(pos,north_pole)*w for pos,w in zip(r['positions'],r['item_weights'])]) + max([haversine(pos,north_pole) for pos in r['positions']])*2*sleigh_weight for r in res]
const_4 = [ sum([haversine(pos,north_pole)*w for pos,w in zip(r['positions'],r['item_weights'])]) + r['pole']*2*sleigh_weight for r in res]

y1 = [r['qopt']-c1 for (r,c1) in zip(res,const_1)]
y2 = [r['qopt']-c2 for (r,c2) in zip(res,const_2)]
y3 = [r['qopt']-c3 for (r,c3) in zip(res,const_3)]
y4 = [r['qopt']-c4 for (r,c4) in zip(res,const_4)]

areas = np.array([r['width']*r['height'] for r in res])
ratios = np.array([r['width']/r['height'] for r in res])


#Interesting: thing, when the constant is c_1, it is better to have large ratios, but when the constant is c_3, low ratios are better !
figure(1);clf()
scat = scatter(areas**0.5,y1,c=ratios)
colorbar(scat, shrink=0.5, aspect=5)

figure(2)
scat = scatter(areas**0.5,y2,c=ratios)
colorbar(scat, shrink=0.5, aspect=5)

figure(3)
scat = scatter(areas**0.5,y3,c=ratios)
colorbar(scat, shrink=0.5, aspect=5)

xx = [a**0.5 for a,r in zip(areas,ratios) if r>0.2]
yy = [y for y,r in zip(y1,ratios) if r>0.2]
scatter(xx,yy)
#in figure1, good fit of best sols with y1 = 2780 * area**0.5
y0 = np.array([y-2780*a**0.5 for y,z in zip(y1,areas)])
figure(4)
scatter(ratios,y0)
fff = lambda (a,b): a*areas**0.5 + b/ratios-y1
scipy.optimize.leastsq(fff,(0,1e4))

fff = lambda (a,b): a*areas**0.5 + b/ratios-y3

#A good fit:
#y1= 2400 area**0.5 + 9140/ratio
figure(5)
scatter(2400 * areas**0.5 + 9140/ratios,y1)
scat = scatter(3179.56 * areas**0.5 + 5570.38/ratios,y1,c=ratios)

figure(5);clf()
scat = scatter(1498 * areas**0.5 -5.57/ratios+1.21e5,y3,c=ratios)

#------------------------------------#
# Result Analysis -- True opt        #
#------------------------------------#



f = open('analysis_statopt_trueopt_3areas','r')
res = eval(f.read())
f.close()

ll=90000
uu=110000
figure(1);clf()
scat = scatter([r['ratio'] for r in res if ll<r['area']<uu],[y1[i] for i,r in enumerate(res) if ll<r['area']<uu])

figure(2);clf()
scat = scatter([r['ratio'] for r in res if ll<r['area']<uu],[y3[i] for i,r in enumerate(res) if ll<r['area']<uu])

figure(3);clf()
scat = scatter([r['ratio'] for r in res if ll<r['area']<uu],[y4[i] for i,r in enumerate(res) if ll<r['area']<uu])


#-------------------------------------#
#-- On a region 10.000 x 600 km     --#
#-------------------------------------#

res = {}
for n in [1,2,3,4,5,6]:#[1,2,3,4,5,6,10,12,15,20,30]:
    J = 60//n
    K = n
    res[n] = {}
    for j in range(J):
        for k in range(K):
            
            width  = 10.*n
            height = 10000./n
                
            top = 45.
            dx = lambda x: haversine((top,0.),(x,0.))-k*height
            lamax = scipy.optimize.fsolve(dx,0)
            dx = lambda x: haversine((top,0.),(x,0.))-(k+1)*height
            lamin = scipy.optimize.fsolve(dx,0)
            dy = lambda y: haversine((top,0.),(top,y))-0.5 * width
            lomin = scipy.optimize.fsolve(dy,-20)
            if lomin>0: lomin = -lomin
            lomax = -lomin
            
            wgt = 0.

            positions = []
            weights = []

            while True:
                wg = random_weight()
                wgt += wg

                if wgt < weight_limit:
                    pos = (lamin + np.random.rand()*(lamax-lamin),lomin + np.random.rand()*(lomax-lomin))
                    pos = (pos[0][0],pos[1][0])
                    positions.append(pos)
                    weights.append(wg)
                else:
                    break

            mlt = subMLT.MLT(None,None,weights,positions)
            res[n][j,k] = mlt.quick_opt().wlatency
            print n,j,k,res[n][j,k]
    print '----'
    print n,sum(res[n].values())
    print '----'
    
'''
[sum(r.values()) for r in res.values()]
Out[352]: 
[653187173.57388115,#1
 630521009.69172549,#2
 624715351.42617869,#3
 626588497.82606816,#4
 630029389.14382505,#5
 628286118.27342618,#6
 638452668.26271176,#10
 641045572.91896641,#12
 642001694.99819899,#15
 644616752.26715946,#20
 648172565.35994923]#30
]


'''