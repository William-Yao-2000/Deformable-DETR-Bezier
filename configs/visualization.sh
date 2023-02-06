# 可视化程序，跑8张图片
startTime=`date +%Y%m%d-%H:%M:%S`
startTime_s=`date +%s`


box_refine_flag=1
modelpath=""
if [ ${box_refine_flag} -eq 1 ]
then
    modelpath="./exps/v002-box-refine/checkpoint0014.pth"
else
    modelpath="./exps/v002/checkpoint0014.pth"
fi
echo "modelpath: ${modelpath}"

imgpath_lst="8/ballet_3_0.jpg     12/batman_3_24.jpg  72/hiking_5_32.jpg  160/stage_4_60.jpg \
            200/zoo_99_96.jpg    200/zoo_99_97.jpg   200/zoo_99_98.jpg   200/zoo_99_99.jpg"

for imgpath in ${imgpath_lst}
do
    echo "\n\n\n----------------${imgpath}----------------"
    imgpath="./data/synthtext/SynthText/${imgpath}"
    if [ ${box_refine_flag} -eq 1 ]
    then  # 加载有 bbox refinement 选项的模型
        python inference_visualization_connect.py \
            --with_box_refine \
            --resume=${modelpath} \
            --inference_img_path=${imgpath}
    else
        python inference_visualization_connect.py \
            --resume=${modelpath} \
            --inference_img_path=${imgpath}
    fi
done


endTime=`date +%Y%m%d-%H:%M:%S`
endTime_s=`date +%s`
 
sumTime=$[ $endTime_s - $startTime_s ]
 
echo "$startTime ---> $endTime" "Total:$sumTime seconds"
