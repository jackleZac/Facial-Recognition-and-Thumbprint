classdef FacialRecognition < matlab.apps.AppBase
    % Properties
    properties (Access = public)
        UIFigure matlab.ui.Figure
        MainPanel matlab.ui.container.Panel
        HeaderPanel matlab.ui.container.Panel
        ControlPanel matlab.ui.container.Panel
        DisplayPanel matlab.ui.container.Panel
        InfoPanel matlab.ui.container.Panel
        TitleLabel matlab.ui.control.Label
        SubtitleLabel matlab.ui.control.Label
        RegisterBtn matlab.ui.control.Button
        ScanBtn matlab.ui.control.Button
        IdentifyBtn matlab.ui.control.Button
        ViewDatabaseBtn matlab.ui.control.Button
        ScanImageAxes matlab.ui.control.UIAxes
        FoundImageAxes matlab.ui.control.UIAxes
        StatusLabel matlab.ui.control.Label
        MatchLabel matlab.ui.control.Label
        ConfidenceLabel matlab.ui.control.Label
        DatabaseCountLabel matlab.ui.control.Label
    end
    
    properties (Access = private)
        faceDetector
        userFaces = {};           % Store original face images
        userFaceFeatures = {};    % Store deep features for matching
        userNames = {};
        scanImage
        scanFeatures              % Store features of scanned image
        dataFile = 'identityData.mat';
        currentViewIndex = 1;
        featureNet                % Deep learning network for features
    end
        
    methods (Access = private)
        
        % ========================================================================
        % DEEP FEATURE EXTRACTION - KEY TO MATCHING DIFFERENT PHOTOS
        % ========================================================================
        
        function features = extractDeepFeatures(app, face)
            % This function extracts deep features that are invariant to:
            % - Different lighting conditions
            % - Different angles (within reason)
            % - Different expressions
            % - Different photo quality
            
            try
                % Resize to network input size
                faceResized = imresize(face, [224 224]);
                
                % Convert to RGB if grayscale
                if size(faceResized, 3) == 1
                    faceResized = cat(3, faceResized, faceResized, faceResized);
                end
                
                % Extract deep features using pre-trained network
                % Using 'fc7' layer which captures high-level facial features
                features = activations(app.featureNet, faceResized, 'fc7', 'OutputAs', 'rows');
                
                % Normalize features for better comparison
                features = features / norm(features);
                
            catch
                % Fallback: Use multiple traditional methods combined
                features = app.extractRobustFeatures(face);
            end
        end
        
        function features = extractRobustFeatures(~, face)
            % Fallback method when deep learning isn't available
            % Combines multiple robust feature extraction methods
            
            if size(face, 3) == 3
                faceGray = rgb2gray(face);
            else
                faceGray = face;
            end
            
            % Resize to standard size
            faceGray = imresize(faceGray, [80 80]);
            
            % Method 1: HOG features (robust to lighting)
            try
                hogFeatures = extractHOGFeatures(faceGray, 'CellSize', [8 8]);
            catch
                hogFeatures = [];
            end
            
            % Method 2: LBP features (robust to illumination)
            try
                lbpFeatures = extractLBPFeatures(faceGray, 'Upright', false);
            catch
                lbpFeatures = [];
            end
            
            % Method 3: SURF features aggregated
            try
                points = detectSURFFeatures(faceGray, 'MetricThreshold', 100);
                [surfFeatures, ~] = extractFeatures(faceGray, points);
                if ~isempty(surfFeatures)
                    % Aggregate SURF features into fixed-length vector
                    surfFeatures = mean(surfFeatures, 1);
                else
                    surfFeatures = zeros(1, 64);
                end
            catch
                surfFeatures = zeros(1, 64);
            end
            
            % Method 4: Histogram features from multiple regions
            % Divide face into grid and extract histograms
            [h, w] = size(faceGray);
            gridSize = 4;
            cellH = floor(h / gridSize);
            cellW = floor(w / gridSize);
            histFeatures = [];
            
            for i = 1:gridSize
                for j = 1:gridSize
                    rowStart = (i-1) * cellH + 1;
                    rowEnd = min(i * cellH, h);
                    colStart = (j-1) * cellW + 1;
                    colEnd = min(j * cellW, w);
                    
                    cell = faceGray(rowStart:rowEnd, colStart:colEnd);
                    hist = imhist(cell);
                    hist = hist / sum(hist); % Normalize
                    histFeatures = [histFeatures; hist];
                end
            end
            
            % Combine all features
            features = [hogFeatures, lbpFeatures, surfFeatures, histFeatures'];
            
            % Normalize
            if ~isempty(features)
                features = features / (norm(features) + eps);
            else
                features = zeros(1, 1000);
            end
        end
        
        function similarity = compareFeatures(~, features1, features2)
            % Compare two feature vectors using multiple similarity metrics
            % No external toolbox dependencies required
            
            if isempty(features1) || isempty(features2)
                similarity = 0;
                return;
            end
            
            % Ensure same length
            minLen = min(length(features1), length(features2));
            features1 = features1(1:minLen);
            features2 = features2(1:minLen);
            
            % Method 1: Cosine similarity (most important for face recognition)
            cosineSim = dot(features1, features2) / (norm(features1) * norm(features2) + eps);
            cosineSim = max(0, cosineSim); % Ensure non-negative
            
            % Method 2: Euclidean distance converted to similarity
            euclidDist = norm(features1 - features2);
            euclidSim = 1 / (1 + euclidDist);
            
            % Method 3: Manual correlation calculation (no toolbox needed)
            % Pearson correlation coefficient formula
            mean1 = mean(features1);
            mean2 = mean(features2);
            
            numerator = sum((features1 - mean1) .* (features2 - mean2));
            denominator = sqrt(sum((features1 - mean1).^2) * sum((features2 - mean2).^2));
            
            if denominator > eps
                corrSim = numerator / denominator;
                corrSim = (corrSim + 1) / 2; % Normalize to 0-1
            else
                corrSim = 0;
            end
            
            % Weighted combination (cosine similarity is most reliable for faces)
            similarity = (cosineSim * 0.7) + (euclidSim * 0.2) + (corrSim * 0.1);
        end
        
        % ========================================================================
        % IMPROVED FACE DETECTION WITH QUALITY VALIDATION
        % ========================================================================
        
        function [face, isValid, qualityReport] = detectAndCropFaceEnhanced(app, img, detector)
            face = [];
            isValid = false;
            qualityReport = struct();
            qualityReport.qualityScore = 0;
            qualityReport.qualityGrade = 'Very Poor';
            qualityReport.error = '';
            
            if size(img, 3) == 1
                imgRGB = cat(3, img, img, img);
            else
                imgRGB = img;
            end
            
            if size(img, 3) == 3
                grayImg = rgb2gray(img);
            else
                grayImg = img;
            end
            
            bbox = step(detector, grayImg);
            
            if isempty(bbox)
                qualityReport.error = 'No face detected in image';
                qualityReport.faceCount = 0;
                return;
            end
            
            numFaces = size(bbox, 1);
            qualityReport.faceCount = numFaces;
            
            if numFaces > 1
                areas = bbox(:, 3) .* bbox(:, 4);
                [~, largestIdx] = max(areas);
                bbox = bbox(largestIdx, :);
                qualityReport.warning = sprintf('%d faces detected, using largest', numFaces);
            end
            
            faceWidth = bbox(3);
            faceHeight = bbox(4);
            minSize = 50;
            
            if faceWidth < minSize || faceHeight < minSize
                qualityReport.error = sprintf('Face too small (%dx%d pixels)', round(faceWidth), round(faceHeight));
                return;
            end
            
            padding = 0.2;
            x = max(1, bbox(1) - bbox(3) * padding);
            y = max(1, bbox(2) - bbox(4) * padding);
            w = min(size(img, 2) - x, bbox(3) * (1 + 2*padding));
            h = min(size(img, 1) - y, bbox(4) * (1 + 2*padding));
            expandedBbox = [x, y, w, h];
            
            face = imcrop(imgRGB, expandedBbox);
            
            blurScore = app.estimateBlur(face);
            qualityReport.blurScore = blurScore;
            
            grayFace = rgb2gray(face);
            brightness = mean(grayFace(:));
            qualityReport.brightness = brightness;
            
            contrast = std(double(grayFace(:)));
            qualityReport.contrast = contrast;
            
            [fh, fw, ~] = size(face);
            qualityReport.resolution = [fh, fw];
            if fh < 80 || fw < 80
                qualityReport.error = 'Face resolution too low for reliable matching';
                return;
            end
            
            try
                faceGray = rgb2gray(face);
                eyeDetector = vision.CascadeObjectDetector('EyePairBig');
                eyeBbox = step(eyeDetector, faceGray);
                qualityReport.eyesDetected = ~isempty(eyeBbox);
            catch
                qualityReport.eyesDetected = false;
            end
            
            qualityScore = 0;
            
            if blurScore >= 200
                qualityScore = qualityScore + 30;
            elseif blurScore >= 100
                qualityScore = qualityScore + 20;
            else
                qualityScore = qualityScore + 10;
            end
            
            if brightness >= 80 && brightness <= 180
                qualityScore = qualityScore + 25;
            elseif brightness >= 60 && brightness <= 200
                qualityScore = qualityScore + 15;
            else
                qualityScore = qualityScore + 5;
            end
            
            if contrast >= 50
                qualityScore = qualityScore + 20;
            elseif contrast >= 30
                qualityScore = qualityScore + 15;
            else
                qualityScore = qualityScore + 5;
            end
            
            if fh >= 150 && fw >= 150
                qualityScore = qualityScore + 15;
            elseif fh >= 100 && fw >= 100
                qualityScore = qualityScore + 10;
            else
                qualityScore = qualityScore + 5;
            end
            
            if qualityReport.eyesDetected
                qualityScore = qualityScore + 10;
            end
            
            qualityReport.qualityScore = qualityScore;
            qualityReport.qualityGrade = app.getQualityGrade(qualityScore);
            
            if qualityScore >= 50  % Lowered threshold since we use deep features
                isValid = true;
                qualityReport.status = 'PASS';
            else
                isValid = false;
                qualityReport.status = 'FAIL';
                qualityReport.error = sprintf('Quality too low (Score: %d/100)', qualityScore);
            end
        end

        function saveData(app)
            userFaces = app.userFaces;
            userNames = app.userNames;
            userFaceFeatures = app.userFaceFeatures;
            save(app.dataFile, 'userFaces', 'userNames', 'userFaceFeatures');
        end

        function loadData(app)
            if isfile(app.dataFile)
                data = load(app.dataFile);
                if isfield(data, 'userFaces')
                    app.userFaces = data.userFaces;
                    app.userNames = data.userNames;
                end
                if isfield(data, 'userFaceFeatures')
                    app.userFaceFeatures = data.userFaceFeatures;
                else
                    app.userFaceFeatures = {};
                end
            end
        end

        function updateDatabaseCount(app)
            count = length(app.userFaces);
            app.DatabaseCountLabel.Text = sprintf('Database: %d identities', count);
        end
        
        function registerNewIdentity(app)
            [file, path] = uigetfile({'*.jpg;*.png;*.jpeg'}, 'Register a new identity');
            if isequal(file, 0)
                return;
            end
            
            try
                img = imread(fullfile(path, file));
                [face, isValid, qualityReport] = app.detectAndCropFaceEnhanced(img, app.faceDetector);
                
                if ~isValid
                    errorMsg = sprintf('Registration failed:\n%s\n\nQuality Score: %d/100 (%s)', ...
                        qualityReport.error, qualityReport.qualityScore, qualityReport.qualityGrade);
                    
                    if isfield(qualityReport, 'blurScore')
                        errorMsg = sprintf('%s\nBlur: %.1f %s', errorMsg, qualityReport.blurScore, ...
                            app.getBlurStatus(qualityReport.blurScore));
                    end
                    
                    uialert(app.UIFigure, errorMsg, 'Quality Check Failed', 'Icon', 'warning');
                    app.StatusLabel.Text = 'Registration failed - Image quality insufficient';
                    app.StatusLabel.FontColor = [1 0.4 0.4];
                    return;
                end
                
                if qualityReport.qualityScore < 70
                    warningMsg = sprintf('Image quality: %s (%d/100)\n%s\n\nContinue with registration?', ...
                        qualityReport.qualityGrade, qualityReport.qualityScore, ...
                        app.getQualityAdvice(qualityReport));
                    
                    choice = uiconfirm(app.UIFigure, warningMsg, 'Quality Warning', ...
                        'Options', {'Register Anyway', 'Cancel'}, ...
                        'DefaultOption', 2, 'Icon', 'warning');
                    
                    if strcmp(choice, 'Cancel')
                        app.StatusLabel.Text = 'Registration cancelled by user';
                        app.StatusLabel.FontColor = [1 0.7 0.3];
                        return;
                    end
                end
                
                % Extract deep features for matching
                faceFeatures = app.extractDeepFeatures(face);
                
                % Enhanced preprocessing for display
                if size(face, 3) == 3
                    faceGray = rgb2gray(face);
                else
                    faceGray = face;
                end
                
                faceEq = adapthisteq(faceGray, 'ClipLimit', 0.02);
                faceEq = wiener2(faceEq, [3 3]);
                faceResized = imresize(faceEq, [80, 80]);
                
                % Store both display image and features
                app.userFaces{end+1} = faceResized;
                app.userFaceFeatures{end+1} = faceFeatures;
                [~, name, ~] = fileparts(file);
                app.userNames{end+1} = name;
                app.saveData();
                app.updateDatabaseCount();
                
                app.StatusLabel.Text = sprintf('✓ Identity registered: %s (Quality: %s)', ...
                    name, qualityReport.qualityGrade);
                app.StatusLabel.FontColor = [0.4 1 0.6];
                
            catch ME
                app.StatusLabel.Text = sprintf('Registration failed: %s', ME.message);
                app.StatusLabel.FontColor = [1 0.4 0.4];
            end
        end

        function scanIdentity(app)
            [file, path] = uigetfile({'*.jpg;*.png;*.jpeg'}, 'Scan identity photo');
            if isequal(file, 0)
                return;
            end
            try
                img = imread(fullfile(path, file));
                [face, isValid, qualityReport] = app.detectAndCropFaceEnhanced(img, app.faceDetector);
                
                if ~isValid
                    errorMsg = sprintf('Scan failed:\n%s\n\nQuality Score: %d/100 (%s)', ...
                        qualityReport.error, qualityReport.qualityScore, qualityReport.qualityGrade);
                    
                    uialert(app.UIFigure, errorMsg, 'Quality Check Failed', 'Icon', 'warning');
                    app.StatusLabel.Text = 'Scan failed - Image quality insufficient';
                    app.StatusLabel.FontColor = [1 0.4 0.4];
                    return;
                end
                
                if qualityReport.qualityScore < 70
                    warningMsg = sprintf('Image quality: %s (%d/100)\n\nProceed with identification?', ...
                        qualityReport.qualityGrade, qualityReport.qualityScore);
                    
                    choice = uiconfirm(app.UIFigure, warningMsg, 'Quality Warning', ...
                        'Options', {'Continue', 'Cancel'}, ...
                        'DefaultOption', 1, 'Icon', 'warning');
                    
                    if strcmp(choice, 'Cancel')
                        app.StatusLabel.Text = 'Scan cancelled by user';
                        app.StatusLabel.FontColor = [1 0.7 0.3];
                        return;
                    end
                end
                
                % Extract features for matching
                app.scanFeatures = app.extractDeepFeatures(face);
                
                % Preprocessing for display
                if size(face, 3) == 3
                    faceGray = rgb2gray(face);
                else
                    faceGray = face;
                end
        
                faceEq = adapthisteq(faceGray, 'ClipLimit', 0.02);
                faceEq = wiener2(faceEq, [3 3]);
                app.scanImage = imresize(faceEq, [80 80]);
        
                imshow(face, 'Parent', app.ScanImageAxes);
                app.ScanImageAxes.XColor = [0.3 0.8 0.5];
                app.ScanImageAxes.YColor = [0.3 0.8 0.5];
                app.ScanImageAxes.LineWidth = 3;
                title(app.ScanImageAxes, sprintf('Scanned Photo (Quality: %s)', qualityReport.qualityGrade), ...
                    'FontSize', 14, 'Color', [0.4 1 0.6]);
                app.StatusLabel.Text = sprintf('Photo scanned - Quality: %s. Ready to identify.', qualityReport.qualityGrade);
                app.StatusLabel.FontColor = [0.4 1 0.6];
        
                app.MatchLabel.Text = '';
                app.ConfidenceLabel.Text = '';
                cla(app.FoundImageAxes);
                title(app.FoundImageAxes, 'Match result', 'FontSize', 14, 'Color', [0.7 0.7 0.7]);
        
            catch ME
                app.StatusLabel.Text = sprintf('Scan failed: %s', ME.message);
                app.StatusLabel.FontColor = [1 0.4 0.4];
            end
        end

        function performIdentification(app)
            if isempty(app.scanImage) || isempty(app.scanFeatures)
                app.StatusLabel.Text = 'Please scan a photo first';
                app.StatusLabel.FontColor = [1 0.7 0.3];
                return;
            end
        
            if isempty(app.userFaces)
                app.StatusLabel.Text = 'No match detected. Database is empty';
                app.StatusLabel.FontColor = [1 0.4 0.4];
                app.MatchLabel.Text = 'NO MATCH DETECTED';
                app.MatchLabel.FontColor = [1 0.4 0.4];
                app.ConfidenceLabel.Text = 'Confidence: 0.0%';
                cla(app.FoundImageAxes);
                title(app.FoundImageAxes, 'Match Result - Empty', 'FontSize', 14, 'Color', [1 0.4 0.4]);
                app.FoundImageAxes.XColor = [1 0.4 0.4];
                app.FoundImageAxes.YColor = [1 0.4 0.4];
                app.FoundImageAxes.LineWidth = 3;
                return;
            end
        
            app.StatusLabel.Text = 'Processing identification using deep features...';
            app.StatusLabel.FontColor = [0.4 0.7 1];
            drawnow;
        
            % Compare using deep features
            similarities = zeros(1, length(app.userFaceFeatures));
            
            for i = 1:length(app.userFaceFeatures)
                similarities(i) = app.compareFeatures(app.scanFeatures, app.userFaceFeatures{i});
            end
            
            [maxSimilarity, bestIdx] = max(similarities);
            
            % CRITICAL FIX: Check if best match is too weak (unknown person)
            % This is the absolute minimum similarity for ANY valid match
            ABSOLUTE_MIN_SIMILARITY = 0.40;  % 40% - adjust based on your testing
            
            if maxSimilarity < ABSOLUTE_MIN_SIMILARITY
                % This person is definitely NOT in the database
                app.MatchLabel.Text = 'NO MATCH DETECTED';
                app.MatchLabel.FontColor = [1 0.4 0.4];
                app.StatusLabel.Text = sprintf('Unknown person (Best match only %.1f%% - too low)', maxSimilarity * 100);
                app.StatusLabel.FontColor = [1 0.4 0.4];
                
                cla(app.FoundImageAxes);
                title(app.FoundImageAxes, 'Unknown Person', 'FontSize', 14, 'Color', [1 0.4 0.4]);
                app.FoundImageAxes.XColor = [1 0.4 0.4];
                app.FoundImageAxes.YColor = [1 0.4 0.4];
                app.FoundImageAxes.LineWidth = 3;
                
                app.ConfidenceLabel.Text = sprintf('All matches below %.0f%% threshold', ABSOLUTE_MIN_SIMILARITY * 100);
                return;
            end
            
            confidence = maxSimilarity * 100;
            
            % Calculate discrimination (separation from second-best)
            sortedSim = sort(similarities, 'descend');
            discrimination = 0;
            
            if length(sortedSim) > 1
                secondBest = sortedSim(2);
                discrimination = (maxSimilarity - secondBest) / (maxSimilarity + eps) * 100;
                
                % Only boost confidence if:
                % 1. There's clear discrimination (>15%)
                % 2. AND the base similarity is already reasonable (>50%)
                if discrimination > 15 && maxSimilarity > 0.50
                    confidence = min(100, confidence * 1.10);  % Reduced boost from 1.15
                end
            end
            
            % Adaptive threshold based on database size and distribution
            baseThreshold = 60;
            
            % Adjust based on database size
            if length(app.userFaces) > 10
                adaptiveThreshold = baseThreshold + 5;
            elseif length(app.userFaces) > 5
                adaptiveThreshold = baseThreshold + 2;
            else
                adaptiveThreshold = baseThreshold;
            end
            
            % Adjust based on confidence distribution
            simStd = std(similarities);
            if simStd < 0.05  % All very similar - be more strict
                adaptiveThreshold = adaptiveThreshold + 10;
            elseif simStd < 0.10
                adaptiveThreshold = adaptiveThreshold + 5;
            end
            
            % Additional check: If discrimination is very low, increase threshold
            if discrimination < 10 && length(sortedSim) > 1
                adaptiveThreshold = adaptiveThreshold + 8;
                app.StatusLabel.Text = 'Warning: Multiple similar matches detected';
            end
            
            % Display results
            if confidence >= adaptiveThreshold
                app.MatchLabel.Text = sprintf('IDENTIFIED: %s', app.userNames{bestIdx});
                app.MatchLabel.FontColor = [0.4 1 0.6];
                
                if confidence >= 85
                    app.StatusLabel.Text = 'Strong match - Very high confidence';
                    app.StatusLabel.FontColor = [0.4 1 0.6];
                elseif confidence >= 70
                    app.StatusLabel.Text = 'Good match - High confidence';
                    app.StatusLabel.FontColor = [0.4 1 0.6];
                elseif confidence >= 60
                    app.StatusLabel.Text = 'Possible match - Moderate confidence (verify recommended)';
                    app.StatusLabel.FontColor = [1 0.9 0.3];
                else
                    app.StatusLabel.Text = 'Weak match - Low confidence (manual verification required)';
                    app.StatusLabel.FontColor = [1 0.7 0.3];
                end
                
                borderColor = [0.3 0.8 0.5];
                
                imshow(app.userFaces{bestIdx}, 'Parent', app.FoundImageAxes);
                app.FoundImageAxes.XColor = borderColor;
                app.FoundImageAxes.YColor = borderColor;
                app.FoundImageAxes.LineWidth = 3;
                title(app.FoundImageAxes, sprintf('Found: %s', app.userNames{bestIdx}), ...
                    'FontSize', 14, 'Color', borderColor);
            else
                app.MatchLabel.Text = 'NO MATCH DETECTED';
                app.MatchLabel.FontColor = [1 0.4 0.4];
                
                if confidence < ABSOLUTE_MIN_SIMILARITY * 100
                    app.StatusLabel.Text = sprintf('Unknown person (Best: %.1f%%)', confidence);
                else
                    app.StatusLabel.Text = sprintf('No confident match (Best: %.1f%% < %.1f%% threshold)', ...
                        confidence, adaptiveThreshold);
                end
                app.StatusLabel.FontColor = [1 0.4 0.4];
                
                cla(app.FoundImageAxes);
                title(app.FoundImageAxes, 'Match Result - No Match', 'FontSize', 14, 'Color', [1 0.4 0.4]);
                app.FoundImageAxes.XColor = [1 0.4 0.4];
                app.FoundImageAxes.YColor = [1 0.4 0.4];
                app.FoundImageAxes.LineWidth = 3;
            end
            
            % Enhanced debug info
            if length(similarities) > 1
                [sortedScores, sortedIdx] = sort(similarities, 'descend');
                debugInfo = 'Top matches: ';
                for k = 1:min(3, length(similarities))
                    debugInfo = sprintf('%s%s(%.1f%%) ', debugInfo, ...
                        app.userNames{sortedIdx(k)}, sortedScores(k)*100);
                end
                app.ConfidenceLabel.Text = sprintf('Confidence: %.1f%% | Threshold: %.1f%% | %s', ...
                    confidence, adaptiveThreshold, debugInfo);
            else
                app.ConfidenceLabel.Text = sprintf('Confidence: %.1f%% | Threshold: %.1f%% | Method: Deep Features', ...
                    confidence, adaptiveThreshold);
            end
        end

        function chi2Dist = chi2Distance(~, h1, h2)
            chi2Dist = 0.5 * sum(((h1 - h2).^2) ./ (h1 + h2 + eps));
            chi2Dist = chi2Dist / length(h1);
        end
        
        function blurScore = estimateBlur(~, img)
            if size(img, 3) == 3
                img = rgb2gray(img);
            end
            laplacian = fspecial('laplacian');
            imgLap = imfilter(img, laplacian, 'replicate');
            blurScore = std2(imgLap)^2;
        end
        
        function grade = getQualityGrade(~, score)
            if score >= 85
                grade = 'Excellent';
            elseif score >= 70
                grade = 'Good';
            elseif score >= 60
                grade = 'Acceptable';
            elseif score >= 40
                grade = 'Poor';
            else
                grade = 'Very Poor';
            end
        end
        
        function status = getBlurStatus(~, score)
            if score >= 200
                status = '(Sharp)';
            elseif score >= 100
                status = '(Acceptable)';
            else
                status = '(Blurry)';
            end
        end
        
        function status = getBrightnessStatus(~, brightness)
            if brightness >= 80 && brightness <= 180
                status = '(Good)';
            elseif brightness < 40
                status = '(Too Dark)';
            elseif brightness > 220
                status = '(Too Bright)';
            else
                status = '(Acceptable)';
            end
        end
        
        function advice = getQualityAdvice(~, report)
            advice = '';
            if report.blurScore < 100
                advice = [advice, '• Use a sharper, focused image', newline];
            end
            if report.brightness < 40
                advice = [advice, '• Image is too dark, use better lighting', newline];
            elseif report.brightness > 220
                advice = [advice, '• Image is overexposed, reduce lighting', newline];
            end
            if report.contrast < 30
                advice = [advice, '• Low contrast, ensure good lighting conditions', newline];
            end
            if isfield(report, 'eyesDetected') && ~report.eyesDetected
                advice = [advice, '• Ensure face is clearly visible and frontal', newline];
            end
            if isempty(advice)
                advice = 'Image meets minimum requirements but could be better';
            end
        end

        function viewDatabase(app)
            if isempty(app.userFaces)
                app.StatusLabel.Text = 'Database is empty';
                app.StatusLabel.FontColor = [1 0.7 0.3];
                return;
            end
            
            if app.currentViewIndex > length(app.userFaces)
                app.currentViewIndex = 1;
            end

            imshow(app.userFaces{app.currentViewIndex}, 'Parent', app.FoundImageAxes);
            app.FoundImageAxes.XColor = [0.4 0.7 1];
            app.FoundImageAxes.YColor = [0.4 0.7 1];
            app.FoundImageAxes.LineWidth = 3;
            title(app.FoundImageAxes, sprintf('Database (%d/%d): %s', ...
                app.currentViewIndex, length(app.userFaces), app.userNames{app.currentViewIndex}), ...
                'FontSize', 14, 'Color', [0.4 0.7 1]);
            app.StatusLabel.Text = sprintf('Viewing database entry %d of %d', app.currentViewIndex, length(app.userFaces));
            app.StatusLabel.FontColor = [0.4 0.7 1];
            
            app.currentViewIndex = app.currentViewIndex + 1;
        end
    end

    % Callbacks
    methods (Access = private)
        function startupFcn(app)
            app.faceDetector = vision.CascadeObjectDetector();
            
            % Try to load pre-trained network for deep features
            try
                app.featureNet = alexnet;
                app.StatusLabel.Text = 'System ready - Using AlexNet for deep features';
            catch
                app.featureNet = [];
                app.StatusLabel.Text = 'System ready - Using traditional features (AlexNet not available)';
            end
            
            app.loadData();
            app.updateDatabaseCount();
            app.StatusLabel.FontColor = [0.4 0.7 1];
        end
        
        function RegisterBtnPushed(app, ~)
            app.registerNewIdentity();
        end
        
        function ScanBtnPushed(app, ~)
            app.scanIdentity();
        end
        
        function IdentifyBtnPushed(app, ~)
            app.performIdentification();
        end
        
        function ViewDatabaseBtnPushed(app, ~)
            app.viewDatabase();
        end
    end

    % Component initialization
    methods (Access = private)
        function createComponents(app)
            % Dark mode color scheme
            darkBg = [0.12 0.12 0.15];
            darkerBg = [0.08 0.08 0.10];
            panelBg = [0.15 0.15 0.18];
            headerBg = [0.10 0.10 0.13];
            accentCyan = [0.3 0.8 0.9];
            accentGreen = [0.4 1 0.6];
            accentRed = [1 0.4 0.4];
            accentPurple = [0.7 0.5 1];
            textColor = [0.9 0.9 0.9];
            
            app.UIFigure = uifigure('Position', [100 100 1000 650]);
            app.UIFigure.Name = 'Facial Recognition System';
            app.UIFigure.Color = darkBg;
            
            app.MainPanel = uipanel(app.UIFigure);
            app.MainPanel.Position = [15 15 970 620];
            app.MainPanel.BackgroundColor = darkerBg;
            app.MainPanel.BorderType = 'none';
            
            app.HeaderPanel = uipanel(app.MainPanel);
            app.HeaderPanel.Position = [20 520 930 80];
            app.HeaderPanel.BackgroundColor = headerBg;
            app.HeaderPanel.BorderType = 'none';
            
            app.TitleLabel = uilabel(app.HeaderPanel);
            app.TitleLabel.Position = [30 35 870 30];
            app.TitleLabel.Text = 'FACIAL RECOGNITION SYSTEM';
            app.TitleLabel.FontSize = 22;
            app.TitleLabel.FontWeight = 'bold';
            app.TitleLabel.FontColor = accentCyan;
            app.TitleLabel.HorizontalAlignment = 'center';
            
            app.SubtitleLabel = uilabel(app.HeaderPanel);
            app.SubtitleLabel.Position = [30 10 870 20];
            app.SubtitleLabel.Text = 'Advanced Deep Learning Identity Recognition & Database Management';
            app.SubtitleLabel.FontSize = 12;
            app.SubtitleLabel.FontColor = [0.6 0.6 0.65];
            app.SubtitleLabel.HorizontalAlignment = 'center';
            
            app.ControlPanel = uipanel(app.MainPanel);
            app.ControlPanel.Position = [20 450 930 60];
            app.ControlPanel.BackgroundColor = panelBg;
            app.ControlPanel.BorderType = 'line';
            app.ControlPanel.BorderWidth = 1;
            app.ControlPanel.HighlightColor = [0.25 0.25 0.28];
            
            app.RegisterBtn = uibutton(app.ControlPanel, 'push');
            app.RegisterBtn.Position = [50 15 160 30];
            app.RegisterBtn.Text = 'Register Identity';
            app.RegisterBtn.FontSize = 11;
            app.RegisterBtn.FontWeight = 'bold';
            app.RegisterBtn.BackgroundColor = [0 0 0];
            app.RegisterBtn.FontColor = [1 1 1];
            app.RegisterBtn.ButtonPushedFcn = createCallbackFcn(app, @RegisterBtnPushed, true);
            
            app.ScanBtn = uibutton(app.ControlPanel, 'push');
            app.ScanBtn.Position = [240 15 160 30];
            app.ScanBtn.Text = 'Scan Photo';
            app.ScanBtn.FontSize = 11;
            app.ScanBtn.FontWeight = 'bold';
            app.ScanBtn.BackgroundColor = [0 0 0];
            app.ScanBtn.FontColor = [1 1 1];
            app.ScanBtn.ButtonPushedFcn = createCallbackFcn(app, @ScanBtnPushed, true);
            
            app.IdentifyBtn = uibutton(app.ControlPanel, 'push');
            app.IdentifyBtn.Position = [430 15 160 30];
            app.IdentifyBtn.Text = 'Identify';
            app.IdentifyBtn.FontSize = 11;
            app.IdentifyBtn.FontWeight = 'bold';
            app.IdentifyBtn.BackgroundColor = [0 0 0];
            app.IdentifyBtn.FontColor = [1 1 1];
            app.IdentifyBtn.ButtonPushedFcn = createCallbackFcn(app, @IdentifyBtnPushed, true);
            
            app.ViewDatabaseBtn = uibutton(app.ControlPanel, 'push');
            app.ViewDatabaseBtn.Position = [620 15 160 30];
            app.ViewDatabaseBtn.Text = 'View Database';
            app.ViewDatabaseBtn.FontSize = 11;
            app.ViewDatabaseBtn.FontWeight = 'bold';
            app.ViewDatabaseBtn.BackgroundColor = [0 0 0];
            app.ViewDatabaseBtn.FontColor = [1 1 1];
            app.ViewDatabaseBtn.ButtonPushedFcn = createCallbackFcn(app, @ViewDatabaseBtnPushed, true);
            
            app.DisplayPanel = uipanel(app.MainPanel);
            app.DisplayPanel.Position = [20 120 930 320];
            app.DisplayPanel.BackgroundColor = panelBg;
            app.DisplayPanel.BorderType = 'line';
            app.DisplayPanel.BorderWidth = 1;
            app.DisplayPanel.HighlightColor = [0.25 0.25 0.28];
            
            app.ScanImageAxes = uiaxes(app.DisplayPanel);
            app.ScanImageAxes.Position = [60 60 380 200];
            app.ScanImageAxes.Color = darkerBg;
            title(app.ScanImageAxes, 'Scanned Photo', 'FontSize', 14, 'Color', textColor);
            app.ScanImageAxes.XTick = [];
            app.ScanImageAxes.YTick = [];
            app.ScanImageAxes.XColor = [0.3 0.3 0.35];
            app.ScanImageAxes.YColor = [0.3 0.3 0.35];
            app.ScanImageAxes.Box = 'on';
            
            app.FoundImageAxes = uiaxes(app.DisplayPanel);
            app.FoundImageAxes.Position = [490 60 380 200];
            app.FoundImageAxes.Color = darkerBg;
            title(app.FoundImageAxes, 'Match Result', 'FontSize', 14, 'Color', textColor);
            app.FoundImageAxes.XTick = [];
            app.FoundImageAxes.YTick = [];
            app.FoundImageAxes.XColor = [0.3 0.3 0.35];
            app.FoundImageAxes.YColor = [0.3 0.3 0.35];
            app.FoundImageAxes.Box = 'on';
            
            app.InfoPanel = uipanel(app.MainPanel);
            app.InfoPanel.Position = [20 20 930 90];
            app.InfoPanel.BackgroundColor = panelBg;
            app.InfoPanel.BorderType = 'line';
            app.InfoPanel.BorderWidth = 1;
            app.InfoPanel.HighlightColor = [0.25 0.25 0.28];
            
            app.StatusLabel = uilabel(app.InfoPanel);
            app.StatusLabel.Position = [30 55 870 20];
            app.StatusLabel.Text = 'System initializing...';
            app.StatusLabel.FontSize = 13;
            app.StatusLabel.FontWeight = 'bold';
            app.StatusLabel.FontColor = textColor;
            
            app.MatchLabel = uilabel(app.InfoPanel);
            app.MatchLabel.Position = [30 30 600 20];
            app.MatchLabel.Text = '';
            app.MatchLabel.FontSize = 15;
            app.MatchLabel.FontWeight = 'bold';
            app.MatchLabel.FontColor = textColor;
            
            app.ConfidenceLabel = uilabel(app.InfoPanel);
            app.ConfidenceLabel.Position = [30 5 400 20];
            app.ConfidenceLabel.Text = '';
            app.ConfidenceLabel.FontSize = 12;
            app.ConfidenceLabel.FontColor = [0.6 0.6 0.65];
            
            app.DatabaseCountLabel = uilabel(app.InfoPanel);
            app.DatabaseCountLabel.Position = [650 5 250 20];
            app.DatabaseCountLabel.Text = 'Database: 0 identities';
            app.DatabaseCountLabel.FontSize = 12;
            app.DatabaseCountLabel.FontColor = accentCyan;
            app.DatabaseCountLabel.HorizontalAlignment = 'right';
        end
    end

    % App creation and deletion
    methods (Access = public)
        function app = FacialRecognition
            createComponents(app)
            registerApp(app, app.UIFigure)
            runStartupFcn(app, @startupFcn)
        end
        
        function delete(app)
            delete(app.UIFigure)
        end
    end
end