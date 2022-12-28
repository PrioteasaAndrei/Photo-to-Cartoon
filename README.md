# Photo to cartoon

## SPG - Tema 2

### Nume: Prioteasa Cristi Andrei
### Grupa: 341 C4

### RULARE : cd src ; python main.py (rularea trebuie facuta din folderul src pentru ca asa e convetia de cale pt imagini)


Pentru a afisa imaginile obtinute pe parcursul prelucrarii pana la forma finala setati SHOW_IMAGES = True (linia 25 din main). Din cauza folosirii unor loopuri peste imagini, majoritatea algoritmilor merg in O(N^2), O(N^3), O(N^2 * logN), deci pentru a obtine toate rezultatele dureaza aproximativ 30 de secunde. Imaginile rezultate sunt salvate in directorul img. Imaginea finala reprezinta o reducere a spatiului culorilor la 16 (prin 4 taieri), 


Imaginea finala reprezinta procesul:


original ->

grayscale ->

gaussian_blur(kernel_size = 9, sigma=1) -> 

sobel_filter(kernel_size=3, cuantizare_gradient=4 unghiuri) ->

non_maximum_supression -> 

double_tresholding -> 

edge_tracking(vicinity=3) -> 

color_quantization(color_pallete=16) ->

median_filter(kernel_size = 5) ->

overlap_images