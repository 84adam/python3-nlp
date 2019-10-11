import numpy as np
from scipy.spatial import distance

# requires equal dimension numpy arrays

def pairwise_jsd(Z):
	Z_dist = distance.pdist(Z, 'jensenshannon')
	return Z_dist

# interpolate missing values from topic probability distributions
# then run pairwise_jsd()

def interp_pairwise_jsd(t_dist):
	
	Z = np.array(t_dist)
	max_len = max([len(x) for x in Z])
	z_topics = []
	
	for i in Z:
		t = [x[0] for x in i]
		z_topics.append(t)
		
	present = []
	
	for i in z_topics:
		for j in i:
			present.append(j)
			
	present = list(set(present))
	
	print("z_topics: ", z_topics)
	print("all topics present: ", present)
	print("interpolating missing topic probabilities...")
	
	fixed_t_dist = []
	
	for x, y in zip(z_topics, t_dist):
		missing = list(set(present).difference(x))
		if len(missing) > 0:
			for i in missing:
				missing_i = (i, 0.0)
				y.append(missing_i)
		fixed_t_dist.append(sorted(y))
	
	for i in fixed_t_dist:
		print(i)
		
	Z2 = np.array(fixed_t_dist)
	
	Z3 = np.array([x[1] for x in fixed_t_dist])
	
	return pairwise_jsd(Z3)
	
# sample distributions

v_A = [(4, 0.7161293), (6, 0.02428288), (15, 0.24149476)]
v_B = [(4, 0.5708665), (9, 0.053239796), (11, 0.024362523), (15, 0.34636793)]
v_C = [(4, 0.14515485), (6, 0.058119595), (9, 0.23094778), (11, 0.5552489)]
v_D = [(2, 0.014619733), (9, 0.38259736), (11, 0.57920164), (18, 0.011559188)]
v_E = [(15, 0.3889571), (18, 0.3415453), (19, 0.24018)]
v_F = [(1, 0.10456503), (2, 0.013440372), (4, 0.23529643), (6, 0.097215556), (15, 0.46182993), (18, 0.084769495)]
v_G = [(9, 0.6103957), (12, 0.05248777), (17, 0.29461083)]
v_H = [(2, 0.058049496), (9, 0.72442913), (17, 0.19779378), (18, 0.014589181)]

t_dist = [v_A, v_B, v_C, v_D, v_E, v_F, v_G, v_H]

interp_pairwise_jsd(t_dist)
