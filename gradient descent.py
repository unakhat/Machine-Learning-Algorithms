ef bivariate_ols(xvalues, yvalues, R=0.01, MaxIterations=1000):
    start_time = time.time()
    epsilon=0.0000001
    iter1=0
    m = xvalues.shape[0] 
    theta = np.ones(xvalues.shape[1])
    x_transpose = xvalues.transpose()
#     print(x_transpose)
#     print(theta)
    hypothesis = np.dot(theta,x_transpose)
    loss = hypothesis - yvalues
    prev_loss=np.sum((np.dot(theta,x_transpose)-yvalues) ** 2) / (2 * m)
    gradient = np.dot(x_transpose, loss) / m
    theta = theta - R * gradient
#     print(prev_loss)
    for iter in range(0, MaxIterations):
        hypothesis = np.dot(theta,x_transpose)
        loss = hypothesis - yvalues
        J = np.sum(loss ** 2) / (2 * m)  # cost
#         print("Iteration %d | Cost: %f" % (iter, J))
        present_loss=J
#         print(prev_loss,present_loss)
        if prev_loss-present_loss<=epsilon:
            break
        else:
            prev_loss=np.sum((np.dot(theta,x_transpose)-yvalues) ** 2) / (2 * m)
        gradient = np.dot(x_transpose, loss) / m         
        theta = theta - R * gradient  # update
        iter1=iter1+1
    print("iter{} | J: {}".format(iter1,J))
    print(theta)
    print("Time taken: {:.2f} seconds".format(time.time() - start_time))
    return theta

        # your code here

# example function call
x=np.asarray(boston['RM'])
y=np.asarray(boston['MEDV'])
x = np.c_[np.ones(x.shape[0]), x]
print("Learning Rate (R)=0.1")
bivariate_ols(x, y, 0.1, 10000000)
print("Learning Rate (R)=0.01")
bivariate_ols(x, y, 0.01, 10000000)
print("Learning Rate (R)=0.001")
bivariate_ols(x, y, 0.001, 10000000)
print("Learning Rate (R)=0.001")
bivariate_ols(x, y, 0.0001, 10000000)
