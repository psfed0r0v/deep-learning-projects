mkdir -p data/
wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz
tar -zxvf edges2shoes.tar.gz
rm edges2shoes.tar.gz
mv edges2shoes data/
wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz
tar -zxvf cityscapes.tar.gz
rm cityscapes.tar.gz
mv cityscapes data/
