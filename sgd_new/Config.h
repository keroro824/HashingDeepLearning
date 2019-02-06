#define ADAM 0
#define BETA1 0.9
#define BETA2 0.999
#define EPS 0.00000001
#define DECAYSTEP 1000
//1: wta; 2: Densified wta; 3: topk minhash; 4: simhash
#define HashFunction 4
//for minhash
#define TOPK 30
//for simhash
#define Ratio 50
//for wta/dwta
#define binsize 8
//Mode 1: hashing; Mode 2: random; Mode 3: topk
#define Mode 4

#define LOADWEIGHT 0

#define MAPLEN 325056

//trainData=/Users/beidchen/Documents/work/SUBLIME/HashingDeepLearning/bgd/q2bn/preproc_tst_0
//testData=/Users/beidchen/Documents/work/SUBLIME/HashingDeepLearning/bgd_new/preproc_tst_shuf_0
//weight=/Users/beidchen/Documents/work/SUBLIME/HashingDeepLearning/bgd_new/test.npz
