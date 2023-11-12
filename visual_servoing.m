% Capture the video frames using the videoinput function
% You have to replace the resolution & your installed adaptor name.
vid = videoinput('winvideo', '1', 'MJPG_640x480');
%vid = webcam(1);
% Set the properties of the video object
set(vid, 'FramesPerTrigger', Inf);
set(vid, 'ReturnedColorspace', 'rgb')
vid.FrameGrabInterval = 2;
start(vid)
x_res = 640;
y_res = 480;
frame_middle = [x_res/2,y_res/2];
detect_bbox_rectangle = [x_res-2,y_res-2,x_res-2,y_res-2];
detect_bbox = reshape(bbox2points(detect_bbox_rectangle)', 1, []);

videoPlayer = vision.VideoPlayer
% Capture one frame to get its size.
%videoFrame = snapshot(vid);
%frameSize = size(videoFrame);
%videoPlayer = vision.VideoPlayer('Position', [200 200 [frameSize(2), frameSize(1)]+30]);

%Init Robot
robot = MyRobot();
robot.move_j(70,-90,-50,-100);
assert(robot.is_robot_connected(),"Robot not connected properly");

while true  
    try
        % Get the snapshot of the current frame
        data = getsnapshot(vid);
        %data = snapshot(vid);
        data2 = data;
        
        % Now to track red objects in real time
        % we have to subtract the red component 
        % from the grayscale image to extract the red components in the image.
        diff_im = imsubtract(data(:,:,1), rgb2gray(data));
        %Use a median filter to filter out noise
        diff_im = medfilt2(diff_im, [3 3]);
        % Convert the resulting grayscale image into a binary image.
        diff_im = im2bw(diff_im,0.25);
        
        % Remove all those pixels less than 300px
        diff_im = bwareaopen(diff_im,300);
        
        % Label all the connected components in the image.
        bw = bwlabel(diff_im, 8);
        
        % Here we do the image blob analysis.
        % We get a set of properties for each labeled region.
        stats = regionprops(bw, 'BoundingBox', 'Centroid');

        %This is a loop to bound the red objects in a rectangular box.
        for object = 1:length(stats)
            bbox = stats(object).BoundingBox;

            if ~isempty(bbox)
                bboxPoints = bbox2points(bbox(1, :));

                bboxPolygon = reshape(bboxPoints', 1, []);
                center = [bbox(1)+bbox(3)/2,bbox(2)+bbox(4)/2];

                % Display a bounding box around the detected face.
                data2 = insertShape(data2, 'Polygon', bboxPolygon, 'LineWidth', 3, 'Color',"blue");
                data2 = insertShape(data2, 'Circle',[frame_middle,5],'LineWidth', 5, 'Color',"red");
                data2 = insertShape(data2, 'Line',[frame_middle,center],'LineWidth', 5, 'Color',"red");
                % delta_j1 = (frame_middle(1)-center(1))/50;
                % delta_j4 = (frame_middle(2)-center(2))/50;
                % 
                % try
                %     robot.move_j(robot.joint_angles(1)+delta_j1,robot.joint_angles(2),robot.joint_angles(3),robot.joint_angles(4));
                %     robot.move_j(robot.joint_angles(1),robot.joint_angles(2),robot.joint_angles(3)+delta_j4,robot.joint_angles(4));
                % catch ME
                %     disp(ME.message);
                % end
            end
        end
        
        % Display the annotated video frame using the video player object.
        step(videoPlayer, data2);
        flushdata(vid);
        % Check whether the video player window has been closed.
        runLoop = isOpen(videoPlayer);
        if ~runLoop
            robot.move_j(0,-90,0,0);
            break
        end
    end
end

% Clean up.
stop(vid);
clearvars -global
release(videoPlayer);
robot.disable_motors();

function dists = get_distances(center, frame_middle)
    x_dist = frame_middle(1) - center(1);
    y_dist = frame_middle(2) - center(2);
    dists = [x_dist,y_dist];
end

function center = get_center(bboxPolygon)
    x = floor((bboxPolygon(1) + bboxPolygon(3)) /2);
    y = floor((bboxPolygon(2) + bboxPolygon(4)) /2);
    center = [x,y];
end

% function depth = get_depth()
%     % Distance: Z = f*B/d %uso la media
%     % Where f is focal length, B is baseline (c1 c2 distance)
%     % d is disparity (horizontal offset of the pointed object)
% 
%     % Given values
%     baselineLength = 0.1; % Baseline length in meters
%     disparity = 20; % Disparity in pixels
%     % Focal length in pixels (or use a real-world value in meters)
%     focalLength_px = 1430;
%     %focalLength_m = 4e-3; %from Logitech HD Webcam C270 
% 
%     % Calculate distance
%     depth = (focalLength_px * baselineLength) / disparity;
% 
%     %This approach assumes that the scene is at the same depth for both 
%     % points. If this assumption does not hold, a more complex calibration 
%     % or depth estimation method might be needed. Also, this is a 
%     % simplified example, and in practice, stereo vision systems are often 
%     % calibrated to obtain more accurate results.
% end

function depth = get_depth(robot, vid)
    
end
