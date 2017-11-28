# Load X and y variable
using JLD
using PyCall
X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
n = size(X,1)

maxDepth = 15
include("decisionTree_infoGain.jl")
Xtest = load("citiesSmall.jld","Xtest")
ytest = load("citiesSmall.jld","ytest")

trainError = []
testError = []

for depth in 1:maxDepth
    model = decisionTree_infoGain(X,y,depth)
    # Evaluate the trianing error
    yhat = model.predict(X)
    push!(trainError, sum(yhat .!= y)/n)

    # Evaluate the test error
    t = size(Xtest, 1)
    yhat = model.predict(Xtest)
    push!(testError, sum(yhat .!= ytest)/t)
end

@pyimport numpy
@pyimport pylab
pylab.plot(1:15, trainError; color="red", linewidth = 2.0, linestyle="--")
pylab.plot(1:15, testError; color="blue", linewidth = 2.0, linestyle="-")
pylab.show()





# Evaluate the test error

t = size(Xtest,1)
yhat = model.predict(Xtest)
testError = sum(yhat .!= ytest)/t
@printf("Test error with depth-%d decision tree: %.3f\n",depth,testError)
