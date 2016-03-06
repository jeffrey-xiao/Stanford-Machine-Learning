function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numFilters,numImages);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

% CONVOLUTION LAYER
value = zeros(convDim, convDim, numFilters, numImages);
conv = zeros(convDim, convDim, numFilters, numImages);

filterArea = filterDim * filterDim;

for imageNum = 1:numImages
  image = images(:, :, imageNum);
  for filterNum = 1:numFilters
    filterTheta = Wc(:, :, filterNum);
    filterTheta = rot90(filterTheta, 2);
     
    value(:, :, filterNum, imageNum) = conv2(image, filterTheta, 'valid') + bc(filterNum);
    conv(:, :, filterNum, imageNum) = sigmoid(value(:, :, filterNum, imageNum));
  end
end

% POOLING LAYER

poolArea = poolDim * poolDim;

for imageNum = 1:numImages
  for filterNum = 1:numFilters
    convLayer = conv(:, :, filterNum, imageNum);
    pooledLayer = conv2(convLayer, ones(poolDim, poolDim) / poolArea, 'valid');
    for x = 1:poolDim:convDim
      for y = 1:poolDim:convDim
        activationsPooled((x - 1) / poolDim + 1, (y - 1) / poolDim + 1, filterNum, imageNum) = pooledLayer(x, y);
      end
    end
  end
end

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses, numImages);

probs = sigmoid(Wd * activationsPooled + bd);

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost
y = speye(10)(labels,:)';

cost = - sum(sum(y .* (log(probs + 1e-8)) + (1 - y) .* (log(1 - probs + 1e-8)))) / numImages;

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

softmaxError = (probs - y) / numImages;

pooledError = (Wd' * softmaxError);
pooledError = reshape(pooledError, outputDim, outputDim, numFilters, numImages);

poolingError = zeros(convDim, convDim, numFilters, numImages);
unpoolingFilter = ones(poolDim, poolDim) / poolArea;

for imageNum=1:numImages
  for filterNum=1:numFilters
    currPoolError = pooledError(:, :, filterNum, imageNum);
    poolingError(:, :, filterNum, imageNum) = kron(currPoolError, unpoolingFilter);
  end
end

convError = poolingError .* sigmoidGrad(value);

%for imageNum=1:numImages
%  for filterNum=1:numFilters
%    currConvError = poolingError(:, :, filterNum, imageNum)
%    convError(:, :, filterNum, imageNum) = currConvError .* sigmoidGrad(value(:, :, imageNum, filterNum));
%  end
%end

%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

Wd_grad = softmaxError * activationsPooled';
bd_grad = sum(softmaxError, 2);

Wc_grad = zeros(size(Wc));
bc_grad = zeros(size(bc));

for filterNum=1:numFilters
  e = convError(:, :, filterNum, :);
  bc_grad(filterNum) = sum(e(:));
end

for filterNum=1:numFilters
  for imageNum=1:numImages
    e = convError(:, :, filterNum, imageNum);
    Wc_grad(:, :, filterNum) += conv2(images(:, :, imageNum), rot90(e, 2), 'valid');
  end
end


%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];
end

function ret = sigmoid(x)
  ret = 1 ./ (1 + exp(-x));
end

function ret  = sigmoidGrad (x)
  ret = sigmoid(x) .* (1 - sigmoid(x));
end