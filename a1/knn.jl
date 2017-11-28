include("misc.jl") # Includes GenericModel typedef


function knn_predict(Xhat,X,y,k)
  (n,d) = size(X)
  (t,d) = size(Xhat)
  k = min(n,k) # To save you some debuggin
  yhat = zeros(t)

  for xh in 1 : t
    dists = zeros(n)
    for x in 1 : n
      dist = sqrt((Xhat[xh, 1] - X[x, 1])^2 + (Xhat[xh, 2] - X[x, 2])^2)
      dists[x] = dist
    end
    ylabels = zeros(k)
    temp = sortperm(dists)

    # reorder the vector
    for l in 1 : k
      ylabels[l] = y[temp[l]]
    end
    yhat[xh] = mode(ylabels)
  end

  return yhat
end

function knn(X,y,k)
	# Implementation of k-nearest neighbour classifier
  predict(Xhat) = knn_predict(Xhat,X,y,k)
  return GenericModel(predict)
end

function cknn(X,y,k)
	# Implementation of condensed k-nearest neighbour classifier
	(n,d) = size(X)
	Xcond = X[1,:]'
	ycond = [y[1]]
	for i in 2:n
    		yhat = knn_predict(X[i,:]',Xcond,ycond,k)
    		if y[i] != yhat[1]
			Xcond = vcat(Xcond,X[i,:]')
			push!(ycond,y[i])
    		end
	end

	predict(Xhat) = knn_predict(Xhat,Xcond,ycond,k)
	return GenericModel(predict)
end
