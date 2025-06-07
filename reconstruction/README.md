## Walk profiles reconstruction
#### Begin
To produce the results of reconstructing walk profiles from powers of the magnetic matrix, we start with running
```
bash run_ip.sh
```  
This will save the reconstruction errors, spectral density into disk (./results/). 
#### Reconstruction
To produce the reconstruction error figures, run
```
python draw_compress.py
```
#### Spectral density
To produce the results of walk profile spectral density, run
```
python draw_spectral_density.py
```
#### Reconstruction vs spectral radius
To study the correlation between reconstruction errors and spectral radius, first compute the spectral radius of the magnetic matrix by running
```
python compute_spectral_radius.py
```
Then run
```
python draw_compress_vs_spectral_radius.py
```
to generate the figures.
