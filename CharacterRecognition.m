function varargout = CharacterRecognition(varargin)
% CHARACTERRECOGNITION MATLAB code for CharacterRecognition.fig
%      CHARACTERRECOGNITION, by itself, creates a new CHARACTERRECOGNITION or raises the existing
%      singleton*.
%
%      H = CHARACTERRECOGNITION returns the handle to a new CHARACTERRECOGNITION or the handle to
%      the existing singleton*.
%
%      CHARACTERRECOGNITION('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CHARACTERRECOGNITION.M with the given input arguments.
%
%      CHARACTERRECOGNITION('Property','Value',...) creates a new CHARACTERRECOGNITION or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CharacterRecognition_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CharacterRecognition_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CharacterRecognition

% Last Modified by GUIDE v2.5 01-Sep-2017 12:42:44

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CharacterRecognition_OpeningFcn, ...
                   'gui_OutputFcn',  @CharacterRecognition_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before CharacterRecognition is made visible.
function CharacterRecognition_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to CharacterRecognition (see VARARGIN)

% Choose default command line output for CharacterRecognition
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes CharacterRecognition wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = CharacterRecognition_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in image_preprocessing.
function image_preprocessing_Callback(hObject, eventdata, handles)
% hObject    handle to image_preprocessing (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
myImage = imread('A.jpg');
set(handles.fig_mouse,'Units','pixels');
resizePos = get(handles.fig_mouse,'Position');
myImage= imresize(myImage, [resizePos(3) resizePos(3)]);
axes(handles.fig_mouse);
imshow(myImage);
set(handles.fig_mouse,'Units','normalized');
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in test.
function test_Callback(hObject, eventdata, handles)
% hObject    handle to test (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in result.
function result_Callback(hObject, eventdata, handles)
% hObject    handle to result (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object creation, after setting all properties.
function fig_mouse_CreateFcn(hObject, eventdata, handles)
% hObject    handle to fig_mouse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
% pos = get(hObject, 'currentpoint'); % get mouse location on figure
% x = pos(1); y = pos(2); % assign locations to x and y

% [a,b]=ginput;
% set(handles.lbl_x, 'string', ['x loc:' num2str(x)]); % update text for x loc
% set(handles.lbl_y, 'string', ['y loc:' num2str(y)]); % update text for y loc 
% Hint: place code in OpeningFcn to populate fig_mouse
function fig_mouse_ButtonDownFcn(hObject, eventdata, handles)
%
%
%

% --- Executes on button press in draw.
function draw_Callback(hObject, eventdata, handles)
% hObject    handle to draw (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
