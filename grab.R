library(mbstructure)
library(igraph)
library(Matrix)

data(MBconnectome)

out <- generate.graph(newrdat, vdf.right, weighted=TRUE) 
g.right <- out$g
vdf.right <- out$vdf
write_graph(g.right, '~/JHU_code/drosophila/right_edges.graphml', "graphml")
write.csv(vdf.right, file='~/JHU_code/drosophila/right_labels.csv')

out <- generate.graph(newldat, vdf.left, weighted=TRUE) 
g.left <- out$g
vdf.left <- out$vdf
write_graph(g.left, '~/JHU_code/drosophila/left_edges.graphml', "graphml")
write.csv(vdf.left, file='~/JHU_code/drosophila/left_labels.csv')

# as_adj_edge_list(g.left)
# is.weighted(g.left)
