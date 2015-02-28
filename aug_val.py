from time import time
begin = time()

from deep.datasets.load import load_plankton
X, y = load_plankton()
X_test, y_test = load_plankton(test=True)

import numpy as np
from deep.augmentation import Reshape
X = Reshape(48).fit_transform(X)
X_test = Reshape(48).fit_transform(X_test)

#: standardize before augmentations (tested and doesn't
#: make a big difference if this is before or after augmentation
#: one thing to check is if we should subtract pixel means
from sklearn.preprocessing import StandardScaler
X = np.vstack((X, X_test))
X = StandardScaler().fit_transform(X)
X_test_batch = X[-len(X_test):]
X = X[:-len(X_test_batch)]

from sklearn.cross_validation import train_test_split
X, X_valid, y, y_valid = train_test_split(X, y, test_size=.1)

#: fixed augmentation for training data
from deep.augmentation.fixed import HorizontalReflection, Rotate
X = np.vstack(HorizontalReflection()(X))
X = np.vstack(Rotate()(X))
y = np.tile(y, 8)

#: shuffle different augmentations
data = zip(X, y)
np.random.shuffle(data)
X, y = zip(*data)

X = np.asarray(X)
y = np.asarray(y)

#: augment valid and reshape to (n_augmentations, n_samples, n_features)
#: we do this since score averages input if ndim == 3
n_valid_samples, n_features = X_valid.shape
X_valid = np.vstack(HorizontalReflection()(X_valid))
X_valid = np.vstack(Rotate()(X_valid))
X_valid = X_valid.reshape((-1, n_valid_samples, n_features))

from deep.layers import Layer, PreConv, Pooling, ConvolutionLayer, PostConv
from deep.activations import RectifiedLinear, Softmax
from deep.initialization import MSR
layers = [
    PreConv(),
    ConvolutionLayer(32, 3, RectifiedLinear(), initialize=MSR()),
    Pooling(2, 2),
    ConvolutionLayer(64, 3, RectifiedLinear(), initialize=MSR()),
    Pooling(2, 2),
    ConvolutionLayer(128, 3, RectifiedLinear(), initialize=MSR()),
    Pooling(2, 2),
    PostConv(),
    Layer(2000, RectifiedLinear(), initialize=MSR()),
    Layer(121, Softmax(), initialize=MSR())
]

from deep.models import NN
from deep.fit import Iterative
from deep.updates import Momentum
from deep.regularizers import L2
nn = NN(layers, .01, Momentum(.9), fit=Iterative(100), regularize=L2(.0005))
nn.fit(X, y, X_valid, y_valid)

#: break test into batch size to avoid memory overflow
#: prediction averages over all types of fixed augmentation
batch_size = 10000
n_test_samples = len(X_test)
n_test_batches = n_test_samples / batch_size
predictions = []
for batch in range(n_test_batches):
    X_test_batch = X_test[batch*batch_size:(batch+1)*batch_size]
    X_test_batch = np.vstack(HorizontalReflection()(X_test_batch))
    X_test_batch = np.vstack(Rotate()(X_test_batch))
    X_test_batch = X_test_batch.reshape((-1, batch_size, n_features))

    from deep.augmentation import RandomPatch
    test_patches = []
    for T in X_test_batch:
        for i in range(3):
            test_patches.append(RandomPatch(48).fit_transform(T))
    X_test_batch = np.asarray(test_patches).reshape((-1, batch_size, 48**2))
    predictions.extend(nn.predict_proba(X_test_batch))

with open('test_submission.csv', 'wb') as submission:
    submission.write('image,acantharia_protist_big_center,acantharia_protist_halo,acantharia_protist,amphipods,appendicularian_fritillaridae,appendicularian_s_shape,appendicularian_slight_curve,appendicularian_straight,artifacts_edge,artifacts,chaetognath_non_sagitta,chaetognath_other,chaetognath_sagitta,chordate_type1,copepod_calanoid_eggs,copepod_calanoid_eucalanus,copepod_calanoid_flatheads,copepod_calanoid_frillyAntennae,copepod_calanoid_large_side_antennatucked,copepod_calanoid_large,copepod_calanoid_octomoms,copepod_calanoid_small_longantennae,copepod_calanoid,copepod_cyclopoid_copilia,copepod_cyclopoid_oithona_eggs,copepod_cyclopoid_oithona,copepod_other,crustacean_other,ctenophore_cestid,ctenophore_cydippid_no_tentacles,ctenophore_cydippid_tentacles,ctenophore_lobate,decapods,detritus_blob,detritus_filamentous,detritus_other,diatom_chain_string,diatom_chain_tube,echinoderm_larva_pluteus_brittlestar,echinoderm_larva_pluteus_early,echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_urchin,echinoderm_larva_seastar_bipinnaria,echinoderm_larva_seastar_brachiolaria,echinoderm_seacucumber_auricularia_larva,echinopluteus,ephyra,euphausiids_young,euphausiids,fecal_pellet,fish_larvae_deep_body,fish_larvae_leptocephali,fish_larvae_medium_body,fish_larvae_myctophids,fish_larvae_thin_body,fish_larvae_very_thin_body,heteropod,hydromedusae_aglaura,hydromedusae_bell_and_tentacles,hydromedusae_h15,hydromedusae_haliscera_small_sideview,hydromedusae_haliscera,hydromedusae_liriope,hydromedusae_narco_dark,hydromedusae_narco_young,hydromedusae_narcomedusae,hydromedusae_other,hydromedusae_partial_dark,hydromedusae_shapeA_sideview_small,hydromedusae_shapeA,hydromedusae_shapeB,hydromedusae_sideview_big,hydromedusae_solmaris,hydromedusae_solmundella,hydromedusae_typeD_bell_and_tentacles,hydromedusae_typeD,hydromedusae_typeE,hydromedusae_typeF,invertebrate_larvae_other_A,invertebrate_larvae_other_B,jellies_tentacles,polychaete,protist_dark_center,protist_fuzzy_olive,protist_noctiluca,protist_other,protist_star,pteropod_butterfly,pteropod_theco_dev_seq,pteropod_triangle,radiolarian_chain,radiolarian_colony,shrimp_caridean,shrimp_sergestidae,shrimp_zoea,shrimp-like_other,siphonophore_calycophoran_abylidae,siphonophore_calycophoran_rocketship_adult,siphonophore_calycophoran_rocketship_young,siphonophore_calycophoran_sphaeronectes_stem,siphonophore_calycophoran_sphaeronectes_young,siphonophore_calycophoran_sphaeronectes,siphonophore_other_parts,siphonophore_partial,siphonophore_physonect_young,siphonophore_physonect,stomatopod,tornaria_acorn_worm_larvae,trichodesmium_bowtie,trichodesmium_multiple,trichodesmium_puff,trichodesmium_tuft,trochophore_larvae,tunicate_doliolid_nurse,tunicate_doliolid,tunicate_partial,tunicate_salp_chains,tunicate_salp,unknown_blobs_and_smudges,unknown_sticks,unknown_unclassified\n')

    for prediction, y in zip(predictions, y_test):
        line = str(y) + ',' + ','.join([str(format(i, 'f')) for i in prediction]) + '\n'
        submission.write(line)
