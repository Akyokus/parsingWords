#include <iostream>
#include <fstream>
#include <string.h>

#include <D:/dlib-19.17/dlib/dnn.h>
#include <D:/dlib-19.17/dlib/data_io.h>
#include <D:/dlib-19.17/dlib/ref.h>

#include <D:/dlib-19.17/dlib/gui_widgets.h>
#include <D:/dlib-19.17/dlib/geometry/rectangle.h>

#include <D:/boost_1_70_0/boost/algorithm/string/split.hpp>


using namespace std;
using namespace dlib;

// A 5x5 conv layer that does 2x downsampling
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
// A 3x3 conv layer that doesn't do any downsampling
template <long num_filters, typename SUBNET> using con3  = con<num_filters,3,3,1,1,SUBNET>;

// Now we can define the 8x downsampling block in terms of conv5d blocks.  We
// also use relu and batch normalization in the standard way.
template <typename SUBNET> using downsampler  = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<32,SUBNET>>>>>>>>>;

// The rest of the network will be 3x3 conv layers with batch normalization and
// relu.  So we define the 3x3 block we will use here.
template <typename SUBNET> using rcon3  = relu<bn_con<con3<32,SUBNET>>>;

// Finally, we define the entire network.   The special input_rgb_image_pyramid
// layer causes the network to operate over a spatial pyramid, making the detector
// scale invariant.
using net_type  = loss_mmod<con<1,6,6,1,1,rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;



template<typename T>

bool existIn(std::vector<T> vec, T item){


    unsigned ite;

    for(ite = 0;ite<vec.size();++ite){
        if(vec[ite] == item){
            return true;
        }
    }

    return false;
}


void thread_load(size_t begin, size_t end, std::vector<string> names, std::vector<matrix<rgb_pixel>> &images){
    string filename;
    interpolate_quadratic interp;
    matrix<rgb_pixel> temp_img, load_img(1144,800);
    cout << "thread started" <<endl;
    for(;begin<end;begin++){
        filename = names[begin];
        load_png(temp_img,filename);
        resize_image(temp_img,load_img,interp);
        images.push_back(load_img);
    }
    cout<< "Images loaded : " << images.size() << endl;

}


int main(int argc, char** argv) try
{

    if(argc !=2){
        cout<< "Please give the location of datas. Or a hint. At least a riddle?" << endl;
        return -1;
    }


    std::vector<matrix<rgb_pixel>> images_train, images_test;
    std::vector<std::vector<mmod_rect>> line_boxes_train, line_boxes_test;

    

    std::vector<string> img_train_filename_vec,img_test_filename_vec;


    std::string dirname = argv[1];


    std::vector<int> test_img_numbers;


    string img_file,img_name_filename = dirname+"/filenames.txt";
    ifstream name_file(img_name_filename);

    cout << "filenames.txt opened"<<endl;

    char charName[50];

    interpolate_quadratic interp;

    matrix<rgb_pixel> temp_img, load_img(1144,800);
    //matrix<unsigned char> gray_img;

    //get 220 rand ones to test and the rest to train

    cout << "Loading images"<<endl;

    cout << "Loading filenames"<<endl;
    //Load filenames to train, and test vectors

    for(int i=0;i<1539;++i){
        name_file.getline(charName,50);

        img_file = std::string(charName);

        int rand_num = std::rand()%1539;

        if(rand_num < 1260){
            //push back to train_images and label
            img_train_filename_vec.push_back(dirname+"/images/"+img_file+".png");
        }else{
            //push back to test_images and label
            test_img_numbers.push_back(i);
            img_test_filename_vec.push_back(dirname+"/images/"+img_file+".png");
        }

    }

    cout << "train filename vec : "<<img_train_filename_vec.size() << " - test filename vec : "<< img_test_filename_vec.size()  <<endl;


    cout << "Loading images"<<endl;

    //
    std::vector<matrix<rgb_pixel>> images_train_first,images_train_second,images_train_third,images_train_forth,images_train_fifth;

    thread_function t1(thread_load,0,250,img_train_filename_vec,dlib::ref(images_train_first));

    thread_function t2(thread_load,250,500,img_train_filename_vec,dlib::ref(images_train_second));

    thread_function t5(thread_load,500,750,img_train_filename_vec,dlib::ref(images_train_third));

    thread_function t6(thread_load,750,1000,img_train_filename_vec,dlib::ref(images_train_forth));


    thread_function t3(thread_load,1000,img_train_filename_vec.size(),img_train_filename_vec,dlib::ref(images_train_fifth));
    
    
    //test images thread
    thread_function t4(thread_load,0,img_test_filename_vec.size(),img_test_filename_vec,dlib::ref(images_test));


    t1.wait();
    t2.wait();
    t3.wait();
    //t5.wait();
    //t6.wait();

    cout << "Threads finished"<<endl;


    for(int i=0;i<images_train_first.size();++i){
        
        images_train.push_back(images_train_first[i]);      
    }
    images_train_first.clear();
    cout << "first erased"<<endl;


    for(int i=0;i<images_train_second.size();++i){
        
        images_train.push_back(images_train_second[i]);      
    }
    images_train_second.clear();
    cout << "second erased"<<endl;

    for(int i=0;i<images_train_third.size();++i){
        
        images_train.push_back(images_train_third[i]);      
    }
    images_train_third.clear();
    cout << "third erased"<<endl;

    for(int i=0;i<images_train_forth.size();++i){
        
        images_train.push_back(images_train_forth[i]);      
    }
    images_train_forth.clear();
    cout << "forth erased"<<endl;

    for(int i=0;i<images_train_fifth.size();++i){
        
        images_train.push_back(images_train_fifth[i]);      
    }
    images_train_fifth.clear();
    cout << "fifth erased"<<endl;



    cout << "Images loaded : "<<images_train.size()<<"  - First image size : " <<images_train[0].size() <<endl;

    cout << "Images loaded"<<endl;



    //LABELS*********************************************************************************

        //Get box data from lines.txt



    string label_filename = dirname + "/labels/lines.txt";

    ifstream lbl_file(label_filename);

    char charLine[200]; //To read lines from .txt file
    string oneLine; //To store the line to be split

    std::vector<std::vector<std::string>> temp_vec;//To stores all splitted strings from the file. Every line is a vector

    cout << "Loading labels"<<endl;

    int k=0;//This is outside while() so that it does not gets in the for loop everytime

    while(!lbl_file.eof()){ // gets the label data into temp_vec[x] = [filename,unused,unused,unused,x_box,y_box,width_box,heigth_box]

        for(; k<23;++k){ //First 23 lines of the file are not used
        lbl_file.getline(charLine,200);
        }


        lbl_file.getline(charLine,200);     //gets the line to charLine
        oneLine = std::string(charLine);    //store it in string so we can split it
        temp_vec.push_back(split(oneLine," ")); //Splits the string by seperator, in this case whitespace="  " and retruns a vector<string> that is pushed back to temp_vec

    }
    cout << "Labels loaded"<<endl;

    int smallest_width=1000, smallest_heigth=1000;
    int biggest_width=0, biggest_heigth=0;

    for(int p=0;p<temp_vec.size()-1;p++){

        if(std::stoi(temp_vec[p][7]) < smallest_width){
            smallest_width = std::stoi(temp_vec[p][7]); 
        }
        if(std::stoi(temp_vec[p][7]) > biggest_width){
            biggest_width = std::stoi(temp_vec[p][7]); 
        }

    }
    for(int p=0;p<temp_vec.size()-1;p++){

    	if(std::stoi(temp_vec[p][6]) < smallest_heigth){
            smallest_heigth = std::stoi(temp_vec[p][7]); 
        }
        if(std::stoi(temp_vec[p][6]) >biggest_heigth){
            biggest_heigth = std::stoi(temp_vec[p][6]); 
        }

    }

    cout << "smallest_width : "<< smallest_width/3.09875 << "  - smallest_heigth : "<< smallest_heigth/3.09875 <<endl;
    cout << "b_width : "<< biggest_width/3.09875 << "  - b_heigth : "<< biggest_heigth/3.09875 <<endl;

    std::vector<mmod_rect> box_vec; //To store box data and save it to line_boxes_train with push.

    cin.get();

    std::string temp_img_name;//used to store the name of current img file

    temp_img_name = temp_vec[0][0]; // Set the name of the first img file so that we can know when we the file changes in temp_vec

    size_t temp_vec_index = 0; //Used to iterate through temp_vec

    rectangle temp_rect(0,0,0,0); // temp rectange to store data and save it to box_vec with push

    cout << "Loading labels to boxes"<<endl;

    for(int line=0; line<1539;++line){ // To get box info of every img
        box_vec.clear();
                                                                                                                                                                         //If this while() works on the first try, I will sacrifice an intern to the C++ gods. First try = (split(temp_img_name,"-")[0] == split(temp_vec[temp_vec_index][0],"-")[0]) &&(split(temp_img_name,"-")[1] == split(temp_vec[temp_vec_index][0],"-")[1])
        while((split(temp_img_name,"-")[0] == split(temp_vec[temp_vec_index][0],"-")[0]) &&(split(temp_img_name,"-")[1] == split(temp_vec[temp_vec_index][0],"-")[1]) ){ //breaks when every box info is saved on a img.So, when current img name is not equal to the name in temp_vec.
            temp_rect.set_left(static_cast<long>((static_cast<double>(std::stoi(temp_vec[temp_vec_index][4])))/3.09875));
            temp_rect.set_top(static_cast<long>((static_cast<double>(std::stoi(temp_vec[temp_vec_index][5])))/3.09875));
            temp_rect.set_right(static_cast<long>((static_cast<double>(std::stoi(temp_vec[temp_vec_index][4])))/3.09875)+static_cast<long>((static_cast<double>(std::stoi(temp_vec[temp_vec_index][6])))/3.09875));
            temp_rect.set_bottom(static_cast<long>((static_cast<double>(std::stoi(temp_vec[temp_vec_index][5])))/3.09875)+static_cast<long>((static_cast<double>(std::stoi(temp_vec[temp_vec_index][7])))/3.09875));
            temp_vec_index++;
            box_vec.push_back(temp_rect);
            if(temp_vec_index == 13353){
                temp_vec_index--;
                break;
            }
        }
        if(existIn(test_img_numbers,line)){
            line_boxes_test.push_back(box_vec);
        }else {
            line_boxes_train.push_back(box_vec);// !!!!!!!!!! This should only happen when this is a train img. To determine search for 'line' in test_img_numbers, if not there do this. Otherwise push it to line_boxes_test
        }

        //cout << "Line : "<< line << " - temp_vec_index : "<< temp_vec_index << " - line_boxes_test : " << line_boxes_test.size() << " - line_boxes_train : " << line_boxes_train.size() <<endl;

        temp_img_name = temp_vec[temp_vec_index][0];  // set the new img name as the current img name
    }
    //LABELS***************************************************************************


    //Testing if data loaded correctly
/*
    save_png(images_train[0],"img.png");
    cout << "img saved"<<endl;
    save_png(images_train[images_train.size()-1],"img2.png");

    cout <<"Left : "<< line_boxes_train[0][0].rect.left()<<endl;

    cout <<"Top : "<< line_boxes_train[0][0].rect.top()<<endl;

    cout <<"Right : "<< line_boxes_train[0][0].rect.right()<<endl;

    cout <<"Bottom : "<< line_boxes_train[0][0].rect.bottom()<<endl;

    image_window win;
    win.set_image(images_train[0]);
    for(auto&& rect : line_boxes_train[0])
        win.add_overlay(rect);
    cin.get();

*/

    //give test num,ask if okay. if not re-do assinging train and test images (or close and open prog)


    mmod_options options(line_boxes_train, 100,38);

    net_type net(options);

    net.subnet().layer_details().set_num_filters(options.detector_windows.size());

    dnn_trainer<net_type> trainer(net);
    trainer.be_verbose();

    trainer.set_learning_rate(0.1);
    trainer.set_learning_rate_shrink_factor(0.1);

    trainer.set_synchronization_file("mmod_sync", std::chrono::minutes(5));

    trainer.set_iterations_without_progress_threshold(1000); // mini-batch size is too low due to hardware limitations, therefore threshhold should be high to prevent premature shrinking. Of the learning rate that is.
    //trainer.set_mini_batch_size(10);

    cout << trainer << endl;


    cout << "starting training"<<endl;
    //trainer.train(images_train,line_boxes_train);

    cin.get();

    random_cropper cropper;
    // We can tell it how big we want the cropped images to be.
    cropper.set_chip_dims(125,700);
    // Also, when doing cropping, it will map the object annotations from the
    // dataset to the cropped image as well as perform random scale jittering.
    // You can tell it how much scale jittering you would like by saying "please
    // make the objects in the crops have a min and max size of such and such".
    // You do that by calling these two functions.  Here we are saying we want the
    // objects in our crops to be no more than 0.8*400 pixels in height and width.
    cropper.set_max_object_size(0.8);
    // And also that they shouldn't be too small. Specifically, each object's smallest
    // dimension (i.e. height or width) should be at least 60 pixels and at least one of
    // the dimensions must be at least 80 pixels.  So the smallest objects the cropper will
    // output will be either 80x60 or 60x80.
    cropper.set_min_object_size(100,38);
    // The cropper can also randomly mirror and rotate crops, which we ask it to
    // perform as well.
    cropper.set_randomly_flip(false);
    cropper.set_max_rotation_degrees(0);
    // This fraction of crops are from random parts of images, rather than being centered
    // on some object.
    cropper.set_background_crops_fraction(0.05);

    // Now ask the cropper to generate a bunch of crops.  The output is stored in
    // crops and crop_boxes.


   std::vector<matrix<rgb_pixel>> mini_batch_samples;
    std::vector<std::vector<mmod_rect>> mini_batch_labels;




    int epoch_count= 0;
    size_t epoch_pos=0;

    size_t mini_batch_size=5; //With 572x400 images and 2GB VRAM mini-batch size can go up to 7.

    while(trainer.get_learning_rate() >= 1e-5){
        
        //used with cropper

        cropper(20, images_train, line_boxes_train, mini_batch_samples, mini_batch_labels); // can go 90

        trainer.train_one_step(mini_batch_samples, mini_batch_labels);


        //To be used without cropper
/*
        for(;epoch_pos<images_train.size();epoch_pos += mini_batch_size){
            trainer.train_one_step(images_train.begin()+epoch_pos,images_train.begin()+std::min(epoch_pos+mini_batch_size,images_train.size()), line_boxes_train.begin()+epoch_pos);

        }

        epoch_pos = 0;
        cout<<"Epoch : " <<epoch_count<<"Leraning rate : " << trainer.get_learning_rate() <<"    Avg. Loss : "<< trainer.get_average_loss() << endl;

        epoch_count++;
*/
    }

    cout << "Training done"<<"   Avg. Loss : "<< trainer.get_average_loss() << endl;
    cin.get();








    net.clean();
    serialize("mmod_network.dat") << net;

    cout << "training results: " << test_object_detection_function(net, images_train, line_boxes_train) << endl;
    // However, to get an idea if it really worked without overfitting we need to run
    // it on images it wasn't trained on.  The next line does this.   Happily,
    // this statement indicates that the detector finds most of the faces in the
    // testing data.
    cout << "testing results:  " << test_object_detection_function(net, images_test, line_boxes_test) << endl;


    // If you are running many experiments, it's also useful to log the settings used
    // during the training experiment.  This statement will print the settings we used to
    // the screen.
    cout << trainer << endl;

    cin.get();

    string img_test_filenums = dirname+"/test_numbers.txt";
    ofstream testnum_file(img_test_filenums);




    

    for(size_t h=0; h<test_img_numbers.size(); ++h)
        testnum_file << test_img_numbers[h] << endl;




}
catch(std::exception& e)
{
    cout << e.what() << endl;
}
