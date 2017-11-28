using JLD
data = load("newsgroups.jld")
X = data["X"]
y = data["y"]
Xtest = data["Xtest"]
ytest = data["ytest"]
wordlist = data["wordlist"]
groupnames = data["groupnames"]


A = y[:,1] .== 1
B = y[:,1] .== 2
C = y[:,1] .== 3
D = y[:,1] .== 4

println(sum(X[A,50]) / sum(A))
println(sum(X[B,50]) / sum(B))
println(sum(X[C,50]) / sum(C))
println(sum(X[D,50]) / sum(D))

println(wordlist[50])

println(X[500,:])

println(wordlist[6])
println(wordlist[24])
println(wordlist[25])
println(wordlist[70])
println(wordlist[88])

println(groupnames[2])
