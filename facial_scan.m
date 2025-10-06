% Biometric GUI: Enrollment and Verification with Robust Camera Image Support

% Create main figure
fig = figure('Name', 'Multimodal Biometric Verification', 'NumberTitle', 'off', ...
    'Position', [100 100 800 600], 'MenuBar', 'none', 'Resize', 'off');

% Axes for displaying images
axFace = axes('Parent', fig, 'Position', [0.05 0.4 0.4 0.5]);
title(axFace, 'Face Image');
axFp = axes('Parent', fig, 'Position', [0.55 0.4 0.4 0.5]);
title(axFp, 'Fingerprint Image');

% Text for status
statusText = uicontrol('Style', 'text', 'Parent', fig, ...
    'Position', [200 50 400 50], 'String', 'Status: Ready', ...
    'FontSize', 12, 'HorizontalAlignment', 'center');

% Store UI components in figure data
data.axFace = axFace;
data.axFp = axFp;
data.statusText = statusText;
guidata(fig, data);

% Button: Enroll
enrollBtn = uicontrol('Style', 'pushbutton', 'Parent', fig, ...
    'Position', [100 150 200 50], 'String', 'Enroll Identity', ...
    'Callback', @enrollCallback);

% Button: Verify
verifyBtn = uicontrol('Style', 'pushbutton', 'Parent', fig, ...
    'Position', [500 150 200 50], 'String', 'Verify Identity', ...
    'Callback', @verifyCallback);

% Enrollment Callback
function enrollCallback(hObject, ~)
    data = guidata(hObject);
    axFace = data.axFace;
    axFp = data.axFp;
    statusText = data.statusText;
    
    try
        set(statusText, 'String', 'Enrolling... Select Face Image');
        [faceFile, facePath] = uigetfile({'*.jpg;*.png', 'Images (*.jpg,*.png)'}, 'Select Face Image');
        if isequal(faceFile, 0)
            set(statusText, 'String', 'Enrollment Canceled');
            return;
        end
        enrollFaceImg = imread(fullfile(facePath, faceFile));
        % Handle orientation and convert to RGB
        info = imfinfo(fullfile(facePath, faceFile));
        if isfield(info, 'Orientation')
            switch info.Orientation
                case 3, enrollFaceImg = imrotate(enrollFaceImg, 180);
                case 6, enrollFaceImg = imrotate(enrollFaceImg, 90);
                case 8, enrollFaceImg = imrotate(enrollFaceImg, -90);
            end
        end
        if ndims(enrollFaceImg) < 3 || size(enrollFaceImg, 3) ~= 3
            enrollFaceImg = cat(3, enrollFaceImg(:,:,1), enrollFaceImg(:,:,1), enrollFaceImg(:,:,1));
        end
        % Resize to manageable size
        enrollFaceImg = imresize(enrollFaceImg, [240 320]);  % Lower resolution
        imshow(enrollFaceImg, 'Parent', axFace);
        title(axFace, 'Enrolled Face');
        
        set(statusText, 'String', 'Select Fingerprint Image');
        [fpFile, fpPath] = uigetfile({'*.jpg;*.png', 'Images (*.jpg,*.png)'}, 'Select Fingerprint Image');
        if isequal(fpFile, 0)
            set(statusText, 'String', 'Enrollment Canceled');
            return;
        end
        enrollFpImg = imread(fullfile(fpPath, fpFile));
        % Handle orientation
        info = imfinfo(fullfile(fpPath, fpFile));
        if isfield(info, 'Orientation')
            switch info.Orientation
                case 3, enrollFpImg = imrotate(enrollFpImg, 180);
                case 6, enrollFpImg = imrotate(enrollFpImg, 90);
                case 8, enrollFpImg = imrotate(enrollFpImg, -90);
            end
        end
        if ndims(enrollFpImg) < 3 || size(enrollFpImg, 3) ~= 3
            enrollFpImg = cat(3, enrollFpImg(:,:,1), enrollFpImg(:,:,1), enrollFpImg(:,:,1));
        end
        enrollFpImg = imresize(enrollFpImg, [240 320]);
        imshow(enrollFpImg, 'Parent', axFp);
        title(axFp, 'Enrolled Fingerprint');
        
        % Very lenient skin detection with debug
        ycbcrEnroll = rgb2ycbcr(enrollFaceImg);
        Cb = ycbcrEnroll(:,:,2);
        Cr = ycbcrEnroll(:,:,3);
        skinMaskEnroll = (Cb > 60 & Cb < 140) & (Cr > 120 & Cr < 190);  % Even wider range
        skinMaskEnroll = imfill(skinMaskEnroll, 'holes');
        skinMaskEnroll = bwareaopen(skinMaskEnroll, 200);  % Very low threshold
        figure('Name', 'Debug Skin Mask'); imshow(skinMaskEnroll);  % Debug window
        statsEnroll = regionprops(skinMaskEnroll, 'Area', 'BoundingBox');
        if ~isempty(statsEnroll)
            [~, maxIdx] = max([statsEnroll.Area]);
            bboxEnroll = statsEnroll(maxIdx).BoundingBox;
            croppedEnrollFace = imcrop(enrollFaceImg, bboxEnroll);
            resizedEnrollFace = imresize(rgb2gray(croppedEnrollFace), [128 128]);
            enrollFaceFeatures = computeHOG(resizedEnrollFace);
        else
            error('No face detected; check lighting or try a closer image');
        end
        
        grayEnrollFp = rgb2gray(enrollFpImg);
        enhancedEnrollFp = adapthisteq(grayEnrollFp);
        binarizedEnrollFp = imbinarize(enhancedEnrollFp);
        thinnedEnrollFp = bwmorph(binarizedEnrollFp, 'thin', Inf);
        enrollFpFeatures = findMinutiae(thinnedEnrollFp);
        
        save('enrolledFeatures.mat', 'enrollFaceFeatures', 'enrollFpFeatures');
        set(statusText, 'String', 'Enrollment Complete');
        msgbox('Identity Enrolled Successfully!', 'Success');
    catch ME
        set(statusText, 'String', ['Error: ' ME.message]);
        msgbox(['Enrollment Failed: ' ME.message], 'Error');
    end
end

% Verification Callback
function verifyCallback(hObject, ~)
    data = guidata(hObject);
    axFace = data.axFace;
    axFp = data.axFp;
    statusText = data.statusText;
    
    try
        if ~exist('enrolledFeatures.mat', 'file')
            msgbox('No identity enrolled yet! Enroll first.', 'Error');
            set(statusText, 'String', 'Error: No Enrolled Identity');
            return;
        end
        load('enrolledFeatures.mat');
        
        set(statusText, 'String', 'Verifying... Select Face Image');
        [faceFile, facePath] = uigetfile({'*.jpg;*.png', 'Images (*.jpg,*.png)'}, 'Select Test Face Image');
        if isequal(faceFile, 0)
            set(statusText, 'String', 'Verification Canceled');
            return;
        end
        testFaceImg = imread(fullfile(facePath, faceFile));
        info = imfinfo(fullfile(facePath, faceFile));
        if isfield(info, 'Orientation')
            switch info.Orientation
                case 3, testFaceImg = imrotate(testFaceImg, 180);
                case 6, testFaceImg = imrotate(testFaceImg, 90);
                case 8, testFaceImg = imrotate(testFaceImg, -90);
            end
        end
        if ndims(testFaceImg) < 3 || size(testFaceImg, 3) ~= 3
            testFaceImg = cat(3, testFaceImg(:,:,1), testFaceImg(:,:,1), testFaceImg(:,:,1));
        end
        testFaceImg = imresize(testFaceImg, [240 320]);
        imshow(testFaceImg, 'Parent', axFace);
        title(axFace, 'Test Face');
        
        set(statusText, 'String', 'Select Fingerprint Image');
        [fpFile, fpPath] = uigetfile({'*.jpg;*.png', 'Images (*.jpg,*.png)'}, 'Select Test Fingerprint Image');
        if isequal(fpFile, 0)
            set(statusText, 'String', 'Verification Canceled');
            return;
        end
        testFpImg = imread(fullfile(fpPath, fpFile));
        info = imfinfo(fullfile(fpPath, fpFile));
        if isfield(info, 'Orientation')
            switch info.Orientation
                case 3, testFpImg = imrotate(testFpImg, 180);
                case 6, testFpImg = imrotate(testFpImg, 90);
                case 8, testFpImg = imrotate(testFpImg, -90);
            end
        end
        if ndims(testFpImg) < 3 || size(testFpImg, 3) ~= 3
            testFpImg = cat(3, testFpImg(:,:,1), testFpImg(:,:,1), testFpImg(:,:,1));
        end
        testFpImg = imresize(testFpImg, [240 320]);
        imshow(testFpImg, 'Parent', axFp);
        title(axFp, 'Test Fingerprint');
        
        testFaceImg = imadjust(testFaceImg);
        testFpImg = medfilt2(rgb2gray(testFpImg), [3 3]);
        
        % Face Pipeline
        ycbcrTest = rgb2ycbcr(testFaceImg);
        Cb = ycbcrTest(:,:,2);
        Cr = ycbcrTest(:,:,3);
        skinMaskTest = (Cb > 60 & Cb < 140) & (Cr > 120 & Cr < 190);
        skinMaskTest = imfill(skinMaskTest, 'holes');
        skinMaskTest = bwareaopen(skinMaskTest, 200);
        statsTest = regionprops(skinMaskTest, 'Area', 'BoundingBox');
        if ~isempty(statsTest)
            [~, maxIdx] = max([statsTest.Area]);
            bboxTest = statsTest(maxIdx).BoundingBox;
            croppedTestFace = imcrop(testFaceImg, bboxTest);
            resizedTestFace = imresize(rgb2gray(croppedTestFace), [256 256]);
            testFaceFeatures = computeHOG(resizedTestFace);
            faceDistance = norm(testFaceFeatures - enrollFaceFeatures);
            faceThreshold = 0.5;
            maxDistance = 1.5;
            faceScore = max(0, 1 - (faceDistance / maxDistance));
            faceMatch = faceDistance < faceThreshold;
        else
            faceScore = 0;
            faceMatch = false;
        end
        
        % Fingerprint Pipeline
        grayTestFp = rgb2gray(testFpImg);
        enhancedTestFp = adapthisteq(grayTestFp);
        binarizedTestFp = imbinarize(enhancedTestFp);
        thinnedTestFp = bwmorph(binarizedTestFp, 'thin', Inf);
        testFpFeatures = findMinutiae(thinnedTestFp);
        fpScore = matchMinutiae(testFpFeatures, enrollFpFeatures);
        fpThreshold = 0.3;
        fpMatch = fpScore > fpThreshold;
        
        % Fusion (Score-Level)
        weightFace = 0.6;
        weightFp = 0.4;
        fusedScore = (weightFace * faceScore) + (weightFp * fpScore);
        fusionThreshold = 0.6;
        verified = fusedScore > fusionThreshold;
        
        if verified
            msgbox(sprintf('Success! Verified (Fused Score: %.2f)', fusedScore), 'Result');
        else
            msgbox(sprintf('Failed! Not Verified (Fused Score: %.2f)', fusedScore), 'Result');
        end
        set(statusText, 'String', 'Verification Complete');
    catch ME
        set(statusText, 'String', ['Error: ' ME.message]);
        msgbox(['Verification Failed: ' ME.message], 'Error');
    end
end

%% Custom Functions
function hogFeatures = computeHOG(img)
    cellSize = 8;
    blockSize = 16;
    numBins = 9;
    img = double(img);
    gx = imfilter(img, [-1 0 1], 'same', 'replicate');
    gy = imfilter(img, [-1; 0; 1], 'same', 'replicate');
    magnitude = sqrt(gx.^2 + gy.^2);
    angle = atan2(gy, gx) * (180 / pi);
    angle(angle < 0) = angle(angle < 0) + 180;
    angle = mod(angle, 180);
    [rows, cols] = size(img);
    numCellsY = floor(rows / cellSize);
    numCellsX = floor(cols / cellSize);
    hist = zeros(numCellsY, numCellsX, numBins);
    binWidth = 180 / numBins;
    for cy = 1:numCellsY
        for cx = 1:numCellsX
            yStart = (cy-1)*cellSize + 1;
            xStart = (cx-1)*cellSize + 1;
            cellMag = magnitude(yStart:yStart+cellSize-1, xStart:xStart+cellSize-1);
            cellAng = angle(yStart:yStart+cellSize-1, xStart:xStart+cellSize-1);
            for i = 1:cellSize
                for j = 1:cellSize
                    ang = cellAng(i,j);
                    mag = cellMag(i,j);
                    bin1 = floor(ang / binWidth) + 1;
                    bin2 = mod(bin1, numBins) + 1;
                    ratio = (ang - (bin1-1)*binWidth) / binWidth;
                    hist(cy, cx, bin1) = hist(cy, cx, bin1) + mag * (1 - ratio);
                    hist(cy, cx, bin2) = hist(cy, cx, bin2) + mag * ratio;
                end
            end
        end
    end
    numBlocksY = floor((rows - blockSize) / cellSize) + 1;
    numBlocksX = floor((cols - blockSize) / cellSize) + 1;
    hogFeatures = [];
    for by = 1:numBlocksY
        for bx = 1:numBlocksX
            blockHist = hist(by:by+1, bx:bx+1, :);
            blockHist = blockHist(:);
            normFactor = sqrt(sum(blockHist.^2) + eps);
            blockHist = blockHist / normFactor;
            hogFeatures = [hogFeatures; blockHist];
        end
    end
end

function minutiae = findMinutiae(thinnedImg)
    [rows, cols] = size(thinnedImg);
    minutiae = [];  % [x, y, type] (1=ending, 3=bifurcation)
    for i = 2:rows-1
        for j = 2:cols-1
            if thinnedImg(i,j) == 1
                neighbors = thinnedImg(i-1:i+1, j-1:j+1);
                neighbors(2,2) = 0;
                cn = 0.5 * sum(abs(diff([neighbors(:); neighbors(1)])));
                if cn == 1
                    minutiae = [minutiae; j, i, 1];
                elseif cn == 3
                    minutiae = [minutiae; j, i, 3];
                end
            end
        end
    end
end

function score = matchMinutiae(testMin, enrollMin)
    if isempty(testMin) || isempty(enrollMin)
        score = 0;
        return;
    end
    
    NI = size(testMin, 1);
    NT = size(enrollMin, 1);
    distThresh = 15;
    scaleSteps = [0.9 1.0 1.1];
    rotSteps = -45:5:45;
    
    centTest = mean(testMin(:,1:2));
    centEnroll = mean(enrollMin(:,1:2));
    maxMatched = 0;
    
    for scale = scaleSteps
        for rot = rotSteps
            rotRad = deg2rad(rot);
            rotMat = [cos(rotRad) -sin(rotRad); sin(rotRad) cos(rotRad)];
            
            scaledTest = (testMin(:,1:2) - centTest) * scale;
            shiftedTest = scaledTest * rotMat + centEnroll;
            
            dists = zeros(NI, NT);
            for i = 1:NI
                for j = 1:NT
                    dx = shiftedTest(i,1) - enrollMin(j,1);
                    dy = shiftedTest(i,2) - enrollMin(j,2);
                    dists(i,j) = sqrt(dx^2 + dy^2);
                end
            end
            
            matched = 0;
            usedEnroll = false(NT, 1);
            for i = 1:NI
                [minDist, idx] = min(dists(i,:));
                if minDist < distThresh && ~usedEnroll(idx) && testMin(i,3) == enrollMin(idx,3)
                    matched = matched + 1;
                    usedEnroll(idx) = true;
                end
            end
            if matched > maxMatched
                maxMatched = matched;
            end
        end
    end
    
    if maxMatched > 0
        score = (maxMatched ^ 2) / (NI * NT);
    else
        score = 0;
    end
    disp(['Matched Minutiae: ' num2str(maxMatched)]);
end