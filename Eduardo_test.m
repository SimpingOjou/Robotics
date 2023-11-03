a = imaqhwinfo;
% [camera_name, camera_id, format] = getCameraInfo(a);
% Capture the video frames using the videoinput function
% You have to replace the resolution & your installed adaptor name.
vid = videoinput('winvideo', '2');
% Set the properties of the video object
set(vid, 'FramesPerTrigger', Inf);
set(vid, 'ReturnedColorspace', 'rgb')
vid.FrameGrabInterval = 1;
%start the video aquisition here
start(vid)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_res = 1024;
y_res = 576;
frame_middle = [x_res/2,y_res/2];
detect_bbox_rectangle = [x_res-2,y_res-2,x_res-2,y_res-2];
detect_bbox = reshape(bbox2points(detect_bbox_rectangle)', 1, []);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

videoPlayer = vision.VideoPlayer

while true
    
    % Get the snapshot of the current frame
    data = getsnapshot(vid);
    data2 = getsnapshot(vid);

    %%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%
    
    % Now to track red objects in real time
    % we have to subtract the red component 
    % from the grayscale image to extract the red components in the image.
    diff_im = imsubtract(data(:,:,1), rgb2gray(data));
    %Use a median filter to filter out noise
    diff_im = medfilt2(diff_im, [3 3]);
    % Convert the resulting grayscale image into a binary image.
    diff_im = im2bw(diff_im,0.18);
    
    % Remove all those pixels less than 300px
    diff_im = bwareaopen(diff_im,300);
    
    % Label all the connected components in the image.
    bw = bwlabel(diff_im, 8);
    
    % Here we do the image blob analysis.
    % We get a set of properties for each labeled region.
    stats = regionprops(bw, 'BoundingBox', 'Centroid');
    
    % Display the image
    %imshow(data)
    
    hold on

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %This is a loop to bound the red objects in a rectangular box.
    for object = 1:length(stats)
        bb = stats(object).BoundingBox;
        bc = stats(object).Centroid;
        rectangle('Position',bb,'EdgeColor','r','LineWidth',2)
        plot(bc(1),bc(2), '-m+')
        a=text(bc(1)+15,bc(2), strcat('X: ', num2str(round(bc(1))), '    Y: ', num2str(round(bc(2)))));
        set(a, 'FontName', 'Arial', 'FontWeight', 'bold', 'FontSize', 12, 'Color', 'yellow');
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        bbox = bb
        bboxPoints = bbox2points(bbox(1, :));

        % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
        % format required by insertShape.

        bboxPolygon = reshape(bboxPoints', 1, []);
        center = [bbox(1)+bbox(3)/2,bbox(2)+bbox(4)/2];

        detect_bbox_color = "green";
        dists = get_distances(center, frame_middle);
        % Display a bounding box around the detected face.
        data2 = insertShape(data2, 'Polygon', bboxPolygon, 'LineWidth', 3, 'Color',"blue");
        data2 = insertShape(data2, 'Circle',[frame_middle,5],'LineWidth', 5, 'Color',"red");
        data2 = insertShape(data2, 'Line',[frame_middle,center],'LineWidth', 5, 'Color',"red");
        delta_j1 = (frame_middle(1)-center(1))/100;
        delta_j4 = (frame_middle(2)-center(2))/100;


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%555
    end
    
    hold off

    step(videoPlayer, data2);%%%%%%%%5
  % Check whether the video player window has been closed.
    runLoop = isOpen(videoPlayer);%%%%%%
    if ~runLoop%%%%%%
        break%%%%%%
    end%%%%%%
    flushdata(vid);

end

% Both the loops end here.
% Stop the video aquisition.
stop(vid);
% Flush all the image data stored in the memory buffer.
flushdata(vid);
% Clear all variables
clear all

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

function inside = face_inside_bbox(center, frame_middle)
    if abs(frame_middle(1) - center(1)) < frame_middle(1) && abs(frame_middle(2) - center(2)) < frame_middle(2)
        inside = 1;
    else
        inside = 0;
    end
end

