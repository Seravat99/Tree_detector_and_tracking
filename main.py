# JUST COMPARE THE TRUNK WHEN IT IS IDENTIFIED IN BOTH FRAMES. IF IT'S NOT IDENTIFIED IN ONE FRAME THEN GIVE IT A VALUE OF 100 WHEN IT DISAPPEARS FROM THE OTHER FRAME

import cv2
import math
import numpy as np
import matplotlib as transforms
import matplotlib.pyplot as plt

#id = 0
#id_follow = 0

# Only the trunks with it's middle on the defined section (a horizontal stripe around the middle of the image, because
# it's looking in height) of the image are considered
def middle_trunks(trunks_with_garbage, height, middle_y):
    section_size = height/9
    if middle_y == 0:
        middle_top = 4*section_size
        middle_bottom = 5*section_size
    else:
        half_section_size = section_size/2
        middle_top = middle_y - half_section_size
        middle_bottom = middle_y + half_section_size
    trunks_without_garbage = []
    for rectangle in trunks_with_garbage:
        middle_height_trunk = rectangle[1] + (rectangle[3]-rectangle[1])/2
        if middle_top < middle_height_trunk < middle_bottom:
            trunks_without_garbage.append(rectangle)
    return trunks_without_garbage


# Check if the trunk is within the median size of the width of the others found and if it is make a new median with that trunk.
def median_width_trunk(frames_buffer, trunks, width):
    trunks_aux = []

    for trunk_id_coord in trunks:
        trunk_width = trunk_id_coord[3] - trunk_id_coord[1]
        if width/5 < trunk_width < 5*width:
            width = (29 * width + trunk_width) / 30
            trunks_aux.append(trunk_id_coord)
        if width == 0:
            width = trunk_width
            trunks_aux.append(trunk_id_coord)

    return width, trunks_aux


# With the black and white filtered image and with the use of histograms, this function finds big enough discrepancies
# of white to black and identifies them as sides of trunks.
# It reads the image from left to right.
# When the left side is found (going up on the histogram), it waits for the right side of the trunk (going down on the
# histogram) before it searches for a new left side of a new trunk.
def find_potential_trunks(dilated_img, height):
    mask = np.uint8(np.where(dilated_img == 0, 1, 0))
    col_counts = cv2.reduce(mask, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
    i = 3
    k = 0
    col_spikes = []
    min_spike = int(col_counts.max() * 0.20)
    # Find the spikes
    for i in range(len(col_counts[0]) - 3):
        # Going up
        if col_counts[0][i + 3] - col_counts[0][i] >= min_spike:
            if k == len(col_spikes):
                col_spikes.append(i)
            else:
                col_spikes[k] = i
            continue
        # Going Down
        if col_counts[0][i - 3] - col_counts[0][i] >= min_spike and len(col_spikes) > k:
            # widens the area found
            trunk_width = i - col_spikes[k]
            if col_spikes[k] - trunk_width > 0:
                col_spikes[k] -= trunk_width
            else:
                col_spikes[k] = 0
            if i + trunk_width < len(col_counts[0]):
                col_spikes.append(i + trunk_width)
            else:
                col_spikes.append(len(col_counts[0]))
            k += 2

    """
    img_copy = img.copy()
    dilated_img_copy = dilated_img.copy()
    # Show histogram
    plt.plot(col_counts[0])
    # plt.show()
    # Put lines in the peaks of the histogram
    for i in col_spikes:
        plt.axvline(i, color='r')
        cv2.line(img_copy, (i, 0), (i, height), (0, 0, 255), 1)
        cv2.line(dilated_img_copy, (i, 0), (i, height), (0, 0, 0), 1)
    plt.title("Number of Black Pixels per Vertical Lines on Image")
    plt.xlabel("Width of Image")
    plt.ylabel("Number of black pixels")
    #plt.show()
    plt.savefig("C:\\Users\\afons\\Documents\\Tese\\Projeto_notNN\\cut_data\\"+str(id)+"_"+str(id_follow)+"_fpt_vertical_lines_plot.png")
    #cv2.imshow("vertical lines", img_copy)
    cv2.imwrite("C:\\Users\\afons\\Documents\\Tese\\Projeto_notNN\\cut_data\\"+str(id)+"_"+str(id_follow)+"_fpt_vertical_lines_img.png", img_copy)
    #cv2.imshow("vertical lines dilated", dilated_img)
    cv2.imwrite("C:\\Users\\afons\\Documents\\Tese\\Projeto_notNN\\cut_data\\"+str(id)+"_"+str(id_follow)+"_fpt_vertical_lines_dilated_img.png", dilated_img_copy)
    plt.close()
    """

    # If the number of spikes of the histogram is odd then take the last one
    if len(col_spikes) % 2 == 1:
        col_spikes.pop()

    # Find the trunks height on the "cut" image
    i = 0
    potential_trunks = []
    while i < len(col_spikes):
        # Ignore the "cut" image if the width is bigger than the height
        if col_spikes[i + 1] - col_spikes[i] <= height:
            # Count black pixels per row after separating the image
            row_counts = cv2.reduce(mask[:, col_spikes[i]:col_spikes[i + 1]], 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)

            top_trunk = 0
            bottom_trunk = 0
            top_trunk_temp = -1
            for j, k in enumerate(row_counts):
                if k == 0 or j == 0 or j == len(row_counts) - 1:
                    if top_trunk_temp == -1:
                        top_trunk_temp = j
                    else:
                        if j - top_trunk_temp > bottom_trunk - top_trunk:
                            top_trunk = top_trunk_temp
                            bottom_trunk = j
                        top_trunk_temp = -1
                        if j < len(row_counts) - 1 and row_counts[j + 1] > 0:
                            top_trunk_temp = j

            """
            # Show on histogram the top and bottom of the trunk
            img_vertical = img[:, col_spikes[i]:col_spikes[i + 1]].copy()
            dilated_img_vertical = dilated_img[:, col_spikes[i]:col_spikes[i + 1]].copy()
            plt.plot(row_counts)
            plt.axvline(bottom_trunk, color='g')
            cv2.line(img_vertical, (0, bottom_trunk), (img_vertical.shape[1], bottom_trunk), (0, 255, 0), 2)
            cv2.line(dilated_img_vertical, (0, bottom_trunk), (img_vertical.shape[1], bottom_trunk), (0, 255, 0), 2)
            plt.axvline(top_trunk, color='g')
            cv2.line(img_vertical, (0, top_trunk), (img_vertical.shape[1], top_trunk), (0, 255, 0), 2)
            cv2.line(dilated_img_vertical, (0, top_trunk), (img_vertical.shape[1], top_trunk), (0, 255, 0), 2)
            plt.title("Number of Black Pixels per Horizontal Line on Vertical Image Cut {}".format(int(i / 2)+1))
            plt.xlabel("Height of Image")
            plt.ylabel("Number of black pixels")
            #plt.show()
            plt.savefig("C:\\Users\\afons\\Documents\\Tese\\Projeto_notNN\\cut_data\\"+str(id)+"_"+str(id_follow)+"_fpt_horizontal_lines_plot"+str(i)+".png")
            #cv2.imshow("horizontal lines "+str(i), img_vertical)
            cv2.imwrite("C:\\Users\\afons\\Documents\\Tese\\Projeto_notNN\\cut_data\\"+str(id)+"_"+str(id_follow)+"_fpt_horizontal_lines_"+str(i)+".png", img_vertical)
            #cv2.imshow("horizontal lines dilated img " + str(i), dilated_img_vertical)
            cv2.imwrite("C:\\Users\\afons\\Documents\\Tese\\Projeto_notNN\\cut_data\\"+str(id)+"_"+str(id_follow)+"_fpt_horizontal_lines_dilated_img_"+str(i)+".png", dilated_img_vertical)
            plt.close()
            """

            # Verify that height is at least 2 times the
            # width
            if bottom_trunk - top_trunk > 2*(col_spikes[i + 1] - col_spikes[i]):
                #cv2.rectangle(img, (col_spikes[i], top_trunk), (col_spikes[i + 1], bottom_trunk), (255, 0, 0), 2)
                potential_trunks.append([col_spikes[i], top_trunk, col_spikes[i + 1], bottom_trunk])

        i += 2  # in order to jump to the next "pair"

    #cv2.imshow("img rectangles", img)
    #cv2.imwrite("C:\\Users\\afons\\Documents\\Tese\\Projeto_notNN\\cut_data\\"+str(id)+"_"+str(id_follow)+"fpt_rectangles.png", img)

    return potential_trunks


# Cover already detected trees with white in the black and white dilated images.
# Cover from top to bottom of the image with the width defined on the function increase_rect_size()
def clean_trees(img, trunks):
    trunks = increase_rect_size(trunks, img.shape)
    for trunk_id_coord in trunks:
        img[:, trunk_id_coord[1]:trunk_id_coord[3]] = [255]

    return img


# Increase the size of the shape found to all sides except if it reaches one of the sides of the image.
def increase_rect_size(trunks, shape):
    height = shape[0]
    width = shape[1]
    update_trunks = []
    for trunk_id_coord in trunks:
        trunk_height = trunk_id_coord[4] - trunk_id_coord[2]
        trunk_width = trunk_id_coord[3] - trunk_id_coord[1]
        trunk_id_coord[2] -= math.ceil(trunk_height * 0.01)
        if trunk_id_coord[2] < 0:  # If the image cuts reach the top of the image
            trunk_id_coord[2] = 0
        trunk_id_coord[4] += math.ceil(trunk_height * 0.01)
        if trunk_id_coord[4] > height:  # If the image cuts reach the bottom of the image
            trunk_id_coord[4] = height

        trunk_id_coord[1] -= math.ceil(trunk_width * 0.3)
        if trunk_id_coord[1] < 0:  # If the image cuts reach the left of the image
            trunk_id_coord[1] = 0
        trunk_id_coord[3] += math.ceil(trunk_width * 0.3)
        if trunk_id_coord[3] > width:  # If the image cuts reach the right of the image
            trunk_id_coord[3] = width
        update_trunks.append(trunk_id_coord)

    return update_trunks


# Defines the id for new trunks and adds it to the beggining of the corresponding trunk list.
# It uses the location where it was found and identifies wether it was on the left or right side of the image, by
# putting an L or an R at the beginning of the ID.
def set_trunk_id(trunks, location, img_width):
    trunks_with_id = []
    left = 0
    right = 0
    img_width_half = img_width / 2
    for trunk in trunks:
        if trunk[0] < img_width_half:
            id = "L" + str(left) + "__" + str(location)
            trunk.insert(0, id)
            trunks_with_id.append(trunk)
            left += 1
        else:
            id = "R" + str(right) + "__" + str(location)
            trunk.insert(0, id)
            trunks_with_id.append(trunk)
            right += 1

    return trunks_with_id


# Compare the two arrays and remove from the first array the rectangles that are overlapping the
# rectangles of the second array.
# Return the first array without the ones removed
def clean_two_arrays(array1, array2):
    if array1 != [] and array2 != []:
        for rect_arr1 in array1:
            for rect_arr2 in array2:
                half_width = (rect_arr2[3] - rect_arr2[1]) / 2
                firstx = rect_arr2[1] - half_width
                secondx = rect_arr2[3] + half_width
                if (firstx <= rect_arr1[0] <= secondx or firstx <= rect_arr1[2] <= secondx or (rect_arr1[0] < firstx and secondx < rect_arr1[2])):
                    array1.remove(rect_arr1)
                    break
    return array1


# Copies a list of lists to a new and independent variable.
def copy_list_of_lists(list1):
    list2 = []
    for sub_list in list1:
        list2.append(sub_list.copy())

    return list2


# If the rectangles of the list are touching the sides (left or right) of the frame, then remove them from the list.
def touch_img_sides(list1, img_width):
    for sub_list in reversed(list1):
        if sub_list[1] <= 0 or sub_list[3] >= (img_width-1):
            list1.remove(sub_list)
    return list1


# This function first checks if there are trunks found in the previous frame in order to track/follow them. It does this
# by cropping the pre processed image with the increased values of the rectangle from the previous frame.
# After updating the coordinates of each rectangle found previously and filling the columns of each rectangle with
# white, it then checks if there are new trunks on the rest of the image. This new trunks/rectangles are given an ID.
def follow(img, trunks, location):
    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    height = img.shape[0]
    width = img.shape[1]
    middle_img = width/2
    new_trunks, update_trunks, new_update_trunks, middle_y = [], [], [], []

    canny_img = cv2.Canny(img, 150, 200)

    # For first row of trees
    dilated_img1 = cv2.dilate(canny_img, SE, iterations=3)

    #global id_follow
    #id_follow = 0
    #img_trunk_updated = img.copy()

    if trunks:
        trunks_copy = copy_list_of_lists(trunks)
        update_trunks = increase_rect_size(trunks_copy, img.shape)
        """
        new_img = img.copy()
        for trunk_id_coord in trunks:
            cv2.rectangle(new_img, [trunk_id_coord[1], trunk_id_coord[2]], [trunk_id_coord[3], trunk_id_coord[4]], (255, 0, 0), 1)
        for trunk_id_coord in update_trunks:
            cv2.rectangle(new_img, [trunk_id_coord[1], trunk_id_coord[2]], [trunk_id_coord[3], trunk_id_coord[4]], (0, 0, 255), 1)
        cv2.imwrite("C:\\Users\\afons\\Documents\\Tese\\Projeto_notNN\\cut_data\\"+str(id)+"box_enlargment.png", new_img)
        """
        for j, trunk_id_coord in enumerate(update_trunks):
            trunk_width = trunks[j][3] - trunks[j][1]
            trunk_height = trunks[j][4] - trunks[j][2]
            trunk_area = trunk_height * trunk_width
            cropped_dilated_img = dilated_img1[trunk_id_coord[2]:trunk_id_coord[4], trunk_id_coord[1]:trunk_id_coord[3]].copy()

            #cropped_img = img[trunk_id_coord[2]:trunk_id_coord[4], trunk_id_coord[1]:trunk_id_coord[3]].copy()

            followed_potential_trunks = find_potential_trunks(cropped_dilated_img, height)

            #id_follow += 1

            pot_trunk_area = 0
            if len(followed_potential_trunks) > 0:
                pot_trunk_index = 0
                for i, pot_trunk in enumerate(followed_potential_trunks):
                    pot_trunk_height = pot_trunk[3] - pot_trunk[1]
                    pot_trunk_width = pot_trunk[2] - pot_trunk[0]
                    pot_trunk_area_aux = pot_trunk_height * pot_trunk_width
                    if pot_trunk_area < pot_trunk_area_aux:
                        pot_trunk_index = i
                        pot_trunk_area = pot_trunk_area_aux
                followed_potential_trunks[pot_trunk_index][0] += trunk_id_coord[1]
                followed_potential_trunks[pot_trunk_index][2] += trunk_id_coord[1]
                followed_potential_trunks[pot_trunk_index][1] += trunk_id_coord[2]
                followed_potential_trunks[pot_trunk_index][3] += trunk_id_coord[2]
                followed_potential_trunks[pot_trunk_index].insert(0, trunk_id_coord[0])
                new_trunks.append(followed_potential_trunks[pot_trunk_index])

                #cv2.rectangle(img_trunk_updated, [followed_potential_trunks[pot_trunk_index][1], followed_potential_trunks[pot_trunk_index][2]], [followed_potential_trunks[pot_trunk_index][3], followed_potential_trunks[pot_trunk_index][4]], (0, 255, 0), 1)
            else:
                old_trunk_id_coord = trunks[j]
                old_trunk_id_coord[2] -= 1
                if old_trunk_id_coord[2] < 0:  # If the image cuts reach the top of the image
                    old_trunk_id_coord[2] = 0
                old_trunk_id_coord[4] += 1
                if old_trunk_id_coord[4] > height:  # If the image cuts reach the bottom of the image
                    old_trunk_id_coord[4] = height
                # Check which side of the image it is, to know if it is for the right or the left to increase
                if old_trunk_id_coord[1] < middle_img and old_trunk_id_coord[3] < middle_img:
                    old_trunk_id_coord[1] -= 1
                    old_trunk_id_coord[3] -= 1
                    if old_trunk_id_coord[1] < 0:  # If the image cuts reach the left of the image
                        old_trunk_id_coord[1] = 0
                elif old_trunk_id_coord[1] >= middle_img and old_trunk_id_coord[3] >= middle_img:
                    old_trunk_id_coord[1] += 1
                    old_trunk_id_coord[3] += 1
                    if old_trunk_id_coord[3] > width:  # If the image cuts reach the right of the image
                        old_trunk_id_coord[3] = width
                new_update_trunks.append(old_trunk_id_coord)

                #cv2.rectangle(img_trunk_updated, [trunks[j][1], trunks[j][2]], [trunks[j][3], trunks[j][4]], (255, 0, 0), 1)
                #cv2.rectangle(img_trunk_updated, [old_trunk_id_coord[1], old_trunk_id_coord[2]], [old_trunk_id_coord[3], old_trunk_id_coord[4]], (0, 0, 255), 1)
        #cv2.imwrite("C:\\Users\\afons\\Documents\\Tese\\Projeto_notNN\\cut_data\\" + str(id) + "box_enlargment_and_displacement.png", img_trunk_updated)

        new_trunks_copy = copy_list_of_lists(new_trunks)
        dilated_img1 = clean_trees(dilated_img1, new_trunks_copy)
        new_update_trunks_copy = copy_list_of_lists(new_update_trunks)
        dilated_img1 = clean_trees(dilated_img1, new_update_trunks_copy)

        #cv2.imwrite("C:\\Users\\afons\\Documents\\Tese\\Projeto_notNN\\cut_data\\" + str(id) + "box_enlargment_and_displacement_dilated.png", dilated_img1)
    potential_trunks = find_potential_trunks(dilated_img1, height)
    trunks_first_row = middle_trunks(potential_trunks, height, 0)
    trunks_first_row = clean_two_arrays(trunks_first_row, new_trunks)
    trunks_first_row = clean_two_arrays(trunks_first_row, new_update_trunks)
    trunks_first_row = set_trunk_id(trunks_first_row, location, width)

    """
    for trunk_id_coord in trunks_first_row:
        cv2.rectangle(img_trunk_updated, [trunk_id_coord[1], trunk_id_coord[2]], [trunk_id_coord[3], trunk_id_coord[4]], (0, 255, 0), 1)
    cv2.imwrite("C:\\Users\\afons\\Documents\\Tese\\Projeto_notNN\\cut_data\\" + str(id) + "all_boxes.png", img_trunk_updated)
    """

    trunks_list = trunks_first_row + new_trunks + new_update_trunks
    trunks_list = touch_img_sides(trunks_list, width)

    return trunks_list


# Ou os troncos encontrados que ficam na buffer sao tracked ou entao têm de ser encontrados de raiz outra vez durante umas 10 frames
# Vou tentar as duas maneiras e depois vejo a mais eficiente
def buffering_begin(trunks_found, buffer_list_begin, frame_width):
    trunks_buffer_begin = []
    number_of_buffer_frames = 10

    for trunk_buffer in reversed(buffer_list_begin):
        trunk_in_list = False
        for trunk_info in trunks_found:
            x_difference = (abs(trunk_info[1] - trunk_buffer[3]) * 100 / frame_width)
            if trunk_info[0][:2] == trunk_buffer[0][:2] and int(trunk_buffer[0][4:]) <= int(trunk_info[0][4:]) <= int(trunk_buffer[0][4:]) + 3 and x_difference <= 2.5:
                trunk_buffer[2] = 0
                trunk_in_list = True
                break
        if not trunk_in_list:
            if trunk_buffer[2] < 3:
                trunk_buffer[2] += 1
            else:
                buffer_list_begin.remove(trunk_buffer)

    for trunk_info in reversed(trunks_found):
        trunk_in_list = False
        for trunk_buffer in reversed(buffer_list_begin):
            x_difference = (abs(trunk_info[1] - trunk_buffer[3]) * 100 / frame_width)
            if trunk_info[0][:2] == trunk_buffer[0][:2] and int(trunk_buffer[0][4:]) <= int(trunk_info[0][4:]) <= int(trunk_buffer[0][4:]) + 3 and x_difference <= 2.5:
                trunk_in_list = True
                trunk_buffer[3] = trunk_info[1]
                if trunk_buffer[1] <= number_of_buffer_frames:
                    trunk_buffer[1] += 1
                    trunk_buffer[0] = trunk_info[0]
                    if trunk_buffer[1] < number_of_buffer_frames:
                        trunks_found.remove(trunk_info)
                break
        if not trunk_in_list:
            buffer_list_begin.append([trunk_info[0], 0, 0, trunk_info[1]])
            trunks_found.remove(trunk_info)

    return trunks_found, buffer_list_begin


# Compares the array of trunks found in the original/clean/old image with the arrays found in the grass grown/unclean/
# new image. If the arrays have similar IDs and rectangle coordinates, then it matches and, in order to know if it needs
# to be cleaned, the difference in area is checked. If the array of the unclean image is empty, but the array of the
# original image is occupied, then it can be assumed that all trees need to be cleaned. The other way around can not happen.
# The percentage of the area of each trunk that needs to be cleaned is updated every time each trunk is detected.
# All trunks detected are saved on the variable "overgrown_tree_location"
def compare_trunks(trunksO, trunksE, overgrown_tree_location, frame_size):
    if len(trunksO) > 0:
        for trunkO_id_coord in trunksO:
            trunkO_height = trunkO_id_coord[4] - trunkO_id_coord[2]
            locO = int(trunkO_id_coord[0][4:])
            right_or_left_O = trunkO_id_coord[0][:2]
            found_tree = False
            # This first part of code whitin the "for" is to add the trunks of the original video to the list or ignore them if they are already there
            for pos, trunk_id_count in enumerate(overgrown_tree_location):
                if trunkO_id_coord[0] == trunk_id_count[0]:
                    otl_pos = pos
                    trunk_id_count[0] = trunkO_id_coord[0]
                    found_tree = True
                    break
            if not(found_tree):
                overgrown_tree_location.append([trunkO_id_coord[0], -1])
                otl_pos = len(overgrown_tree_location)-1

            if len(trunksE) > 0:
                found_tree_in_E = False
                for trunkE_id_coord in trunksE:
                    locE = int(trunkE_id_coord[0][4:])
                    right_or_left_E = trunkE_id_coord[0][:2]
                    if right_or_left_E == right_or_left_O:
                        x_difference = int(abs(trunkO_id_coord[1] - trunkE_id_coord[1]) * 100 / frame_size[0])
                        y_difference = int(abs(trunkO_id_coord[2] - trunkE_id_coord[2]) * 100 / frame_size[1])
                        if x_difference <= 5 and y_difference <= 5:  # CHECK if the difference is bigger than 5% of the frame size
                            trunkE_height = trunkE_id_coord[4] - trunkE_id_coord[2]
                            height_perc = 100 - int(trunkE_height * 100 / trunkO_height)
                            if overgrown_tree_location[otl_pos][1] == -1:
                                overgrown_tree_location[otl_pos][1] = abs(height_perc)
                            else:
                                overgrown_tree_location[otl_pos][1] = ((overgrown_tree_location[otl_pos][1] * 29 + height_perc) / 30)
                            found_tree_in_E = True
                            break
                if not (found_tree_in_E):
                    continue

    return overgrown_tree_location


# Receives the videos, resizes each frame to (854, 480) and calls the compare_trunks() function in order to compare the
# just resized frames of the two videos. After making the comparisons and receiving the values of each tree, it then
# calls the buffering_end() function twice, one for each frame, (??and then writes on the created video the new frame with
# rectangles??). At the end of the program it tells which trees need to be cleaned.
if __name__ == '__main__':
    img_video = '1'  # input("image - 0 ; video - 1: ")
    if img_video == '0':
        imgO = cv2.imread('C:\\Users\\afons\\Documents\\Tese\\Projeto_notNN\\data\\WalnutAcres2_11.png')
        imgE = cv2.imread('C:\\Users\\afons\\Documents\\Tese\\Projeto_notNN\\data\\WalnutAcres2_0_e.png')
        imgO = cv2.resize(imgO, (854, 480))
        imgE = cv2.resize(imgE, (854, 480))
        cv2.imshow("Original img", imgO)
        cv2.imshow("Edited img", imgE)
        #trunks_list, trunks_first_row, new_trunks, new_update_trunks = follow(imgO, [], 1)
        #imgO_e, imgE_e, trunksO, trunksE = compare_trunks(imgO, imgE, [],[])
        #cv2.imshow("Original edited img", imgO)

    elif img_video == '1':
        video_directory0 = "C:\\Users\\afons\\Documents\\Tese\\Projeto_notNN\\data\\WalnutAcres2.mp4"
        video_directory1 = "C:\\Users\\afons\\Documents\\Tese\\Projeto_notNN\\data\\WalnutAcres2_eee.mp4"
        cap0 = cv2.VideoCapture(video_directory0)
        cap1 = cv2.VideoCapture(video_directory1)
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0:
            print("No clean video to read")
        else:
            if not ret1:
                print("No grown video to read")
            else:
                frame_width = 854
                frame_height = 480
                frame_size = (frame_width, frame_height)
                frame0 = cv2.resize(frame0, frame_size)
                frame1 = cv2.resize(frame1, frame_size)
                #video0 = cv2.VideoWriter('video with trunks clean.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, frame_size)
                #video0.write(frame0)
                #video1 = cv2.VideoWriter('video with trunks grown.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, frame_size)
                #video1.write(frame1)
                trunksO_temp, trunksE_temp = [], []
                trunksO_found, trunksE_found = [], []
                buffer_list_beginO, buffer_list_beginE = [], []
                first_buf0, first_buf1, second_buf0, second_buf1 = [], [], [], []
                grown_tree_location = []
                areaO, areaE = 0, 0
                widthO, widthE = 0, 0
                frames_buffer = 0
                location0, location1 = 0, 0
                while True:
                    # print("---------------------------------------------")
                    location0 = frames_buffer
                    location1 = frames_buffer
                    ret0, frame0 = cap0.read()
                    ret1, frame1 = cap1.read()
                    if not ret1 or not ret0:
                        break
                    frame0 = cv2.resize(frame0, frame_size)
                    frame1 = cv2.resize(frame1, frame_size)

                    trunksO, trunksO_e, trunksO_aux, found_trunksO, followed_trunksO, buffering_trunksO, trunksO_buffer_begin, trunksO_tracked, trunksO_aux_buffer = [], [], [], [], [], [], [], [], []
                    trunksE, trunksE_e, trunksE_aux, found_trunksE, followed_trunksE, buffering_trunksE, trunksE_buffer_begin, trunksE_tracked, trunksE_aux_buffer = [], [], [], [], [], [], [], [], []

                    trunksO_aux = follow(frame0.copy(), trunksO_temp, location0)
                    trunksO_aux_buffer = copy_list_of_lists(trunksO_aux)
                    trunksO_aux_buffer, buffer_list_beginO = buffering_begin(trunksO_aux_buffer, buffer_list_beginO, frame_width)
                    widthO, trunksO = median_width_trunk(frames_buffer, trunksO_aux_buffer, widthO)
                    """
                    for trunk_id_coord in trunksO:
                        cv2.rectangle(frame0, [trunk_id_coord[1], trunk_id_coord[2]], [trunk_id_coord[3], trunk_id_coord[4]], (0, 0, 255), 1)
                        cv2.putText(frame0, trunk_id_coord[0], [trunk_id_coord[1], trunk_id_coord[2]], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    """
                    trunksE_aux = follow(frame1.copy(), trunksE_temp, location1)
                    trunksE_aux_buffer = copy_list_of_lists(trunksE_aux)
                    trunksE_aux_buffer, buffer_list_beginE = buffering_begin(trunksE_aux_buffer, buffer_list_beginE, frame_width)
                    widthE, trunksE = median_width_trunk(frames_buffer, trunksE_aux_buffer, widthE)
                    """
                    for trunk_id_coord in trunksE:
                        cv2.rectangle(frame1, [trunk_id_coord[1], trunk_id_coord[2]], [trunk_id_coord[3], trunk_id_coord[4]], (0, 0, 255), 1)
                        cv2.putText(frame1, trunk_id_coord[0], [trunk_id_coord[1], trunk_id_coord[2]], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    """
                    grown_tree_location = compare_trunks(trunksO, trunksE, grown_tree_location, frame_size)

                    trunksO_temp = trunksO_aux_buffer
                    trunksE_temp = trunksE_aux_buffer

                    #video0.write(frame0)
                    #video1.write(frame1)

                    frames_buffer += 1
                    #id += 1
                #video0.release()
                #video1.release()

                if grown_tree_location == []:
                    print("Something went wrong")
                else:
                    for trunk_id_count in grown_tree_location:
                        trunk_id_count[1] = round(trunk_id_count[1])

                        if trunk_id_count[0][:1] == 'L':
                            right_or_left = 'left'
                        else:
                            right_or_left = 'right'

                        total_seconds = int(int(trunk_id_count[0][4:]) / 30)
                        minutes = int(total_seconds / 60)
                        seconds = round(total_seconds % 60)

                        if trunk_id_count[1] == -1:
                            trunk_id_count[1] = 100
                        if trunk_id_count[1] >= 20:
                            if seconds > 9:
                                print("The tree that was first found at {}:{} on the {} needs to be checked".format(minutes, seconds, right_or_left))
                            else:
                                print("The tree that was first found at {}:0{} on the {} needs to be checked".format(minutes, seconds, right_or_left))
                        elif trunk_id_count[1] < 0:
                            if seconds > 9:
                                print("The \"reference\" video needs to be checked. The tree that was first found at {}:{} on the {} was bigger on the \"new/overgrown\" video.".format(minutes, seconds, right_or_left))
                            else:
                                print("The \"reference\" video needs to be checked. The tree that was first found at {}:0{} on the {} was bigger on the \"new/overgrown\" video.".format(minutes, seconds, right_or_left))
                    for trunk_id_count in grown_tree_location:
                        print(trunk_id_count)

        print("End")

    cv2.waitKey(0)
    cv2.destroyAllWindows()



#
# # Check if the trunk is within the median size of the others found and if it is make a new median with that trunk.
# def median_size_trunk(frames_buffer, trunks, area):
#     trunks_aux = []
#
#     for trunk_id_coord in trunks:
#         trunk_height = trunk_id_coord[4] - trunk_id_coord[2]
#         trunk_width = trunk_id_coord[3] - trunk_id_coord[1]
#         trunk_area = trunk_height * trunk_width
#         if area/6 < trunk_area < 5*area:
#             area = (29 * area + trunk_area) / 30
#             trunks_aux.append(trunk_id_coord)
#         if area == 0:
#             area = trunk_area
#             trunks_aux.append(trunk_id_coord)
#
#     return area, trunks_aux
#
#
# Check if the width of the rectangles closer to the left and right edges of the image are bigger than the rectangles
# closer to the center of the image.
# def extremes_bigger(list1):
#     left_width = 0
#     left_list = ['X', 1000000, 0, 0, 0]
#     right_width = 0
#     right_list = ['X', 0, 0, 0, 0]
#
#     for sub_list in reversed(list1):
#         sub_list_width = (sub_list[3] - sub_list[1])
#         if sub_list[0][:1] == 'L':
#             if sub_list_width > left_width:
#                 if sub_list[1] > left_list[1]:
#                     if int(sub_list[0][4:]) <= int(left_list[0][4:]):
#                         list1.remove(left_list)
#                     else:
#                         list1.remove(sub_list)
#                 if left_list[0] == 'X' or int(sub_list[0][4:]) <= int(left_list[0][4:]):
#                     left_list = sub_list
#                     left_width = sub_list_width
#             elif sub_list[1] < left_list[1]:
#                 if int(left_list[0][4:]) <= int(sub_list[0][4:]):
#                     list1.remove(sub_list)
#                 else:
#                     list1.remove(left_list)
#         else:
#             if sub_list_width > right_width:
#                 if sub_list[1] < right_list[1]:
#                     if int(sub_list[0][4:]) <= int(right_list[0][4:]):
#                         list1.remove(right_list)
#                     else:
#                         list1.remove(sub_list)
#                 if right_list[0] == 'X' or int(sub_list[0][4:]) <= int(right_list[0][4:]):
#                     right_list = sub_list
#                     right_width = sub_list_width
#             elif sub_list[1] > right_list[1]:
#                 if int(right_list[0][4:]) <= int(sub_list[0][4:]):
#                     list1.remove(sub_list)
#                 else:
#                     list1.remove(right_list)
#
#     return list1
#
#
# In case the trunk is being detected but is lost, this function helps to keep it until it is found again or, if it is
# not found in the next x frames ignore it.
# def buffering_end(buffer_list_end, first_buf, second_buf):
#     for second_buf_trunk in second_buf:
#         if second_buf_trunk in buffer_list_end:
#             buffer_list_end.remove(second_buf_trunk)
#     second_buf = []
#     for first_buf_trunk in first_buf:
#         if first_buf_trunk in buffer_list_end:
#             buffer_list_end.remove(first_buf_trunk)
#             second_buf.append(first_buf_trunk)
#     first_buf = buffer_list_end
#
#     return buffer_list_end, first_buf, second_buf
#
#
# Second part of the follow function that was to follow and find rectangles more on the middle of the image
# """
#     canny_img = clean_trees(canny_img, trunks_first_row)
#     # For second row of trees
#     dilated_img2 = cv2.dilate(canny_img, SE, iterations=2)
#     if new_update_trunks:
#         new_trunks = []
#         for trunk in new_update_trunks:
#             cropped_img = dilated_img2[trunk[0][1]:trunk[1][1], trunk[0][0]:trunk[1][0]].copy()
#             potential_trunks = find_potential_trunks(cropped_img, height)
#             if len(potential_trunks) > 0:
#                 pot_trunk_area = 0
#                 pot_trunk_index = 0
#                 for i, pot_trunk in enumerate(potential_trunks):
#                     pot_trunk_height = pot_trunk[1][1] - pot_trunk[0][1]
#                     pot_trunk_width = pot_trunk[1][0] - pot_trunk[0][0]
#                     pot_trunk_area_aux = pot_trunk_height * pot_trunk_width
#                     if pot_trunk_area < pot_trunk_area_aux:
#                         pot_trunk_index = i
#                 potential_trunks[pot_trunk_index][0][0] += trunk[0][0]
#                 potential_trunks[pot_trunk_index][1][0] += trunk[0][0]
#                 potential_trunks[pot_trunk_index][0][1] += trunk[0][1]
#                 potential_trunks[pot_trunk_index][1][1] += trunk[0][1]
#                 new_trunks += [potential_trunks[pot_trunk_index]]
#             # else:
#         dilated_img2 = clean_trees(dilated_img2, new_trunks)
#         trunks_first_row += new_trunks
#     dilated_img2, middle_y = clean_for_second_row(dilated_img2, trunks_first_row)
#     potential_trunks = find_potential_trunks(dilated_img2, height)
#     trunks_second_row = middle_trunks(potential_trunks, height, [])
#     """
#
#
# def extremes_bigger_v1(list1):
#     left_pos, left_area, left_index = 1000, 0, 0
#     right_pos, right_area, right_index = 0, 0, 0
#     list2 = copy_list_of_lists(list1)
#     for i, sub_list in enumerate(list1):
#         sub_list_area = (sub_list[3] - sub_list[1]) * (sub_list[4] - sub_list[2])
#         if sub_list[0][:1] == 'L':
#             if sub_list_area > left_area:
#                 if sub_list[1] > left_pos:
#                     list2.pop(left_index)
#                     left_index = i-1
#                 left_pos = sub_list[1]
#                 left_area = sub_list_area
#                 left_index = i
#             if sub_list_area < left_area and sub_list[1] < left_pos:
#                 list2.remove(sub_list)
#         else:
#             if sub_list_area > right_area:
#                 if sub_list[1] < right_pos:
#                     list2.pop(right_index)
#                     right_index = i - 1
#                 right_pos = sub_list[1]
#                 right_area = sub_list_area
#                 right_index = i
#             if sub_list_area < right_area and sub_list[1] > right_pos:
#                 list2.remove(sub_list)
#
#     return list2
#
#
#
# def follow_trunks(img, trunks):
#     SE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     dMat = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
#     height = img.shape[0]
#     width = img.shape[1]
#     middle_img = width/2
#     new_trunks, update_trunks, middle_y = [], [], []
#
#     # Increase the rectangle of the image cut to the side the tree went and up and down
#     for trunk in trunks:
#         trunk_height = trunk[1][1] - trunk[0][1]
#         trunk_width = trunk[1][0] - trunk[0][0]
#         trunk[0][1] -= math.ceil(trunk_height / 100)
#         if trunk[0][1] < 0: # If the image cuts reach the top of the image
#             trunk[0][1] = 0
#         trunk[1][1] += math.ceil(trunk_height / 100)
#         if trunk[1][1] > height: # If the image cuts reach the bottom of the image
#             trunk[1][1] = height
#         # Check which side of the image it is, to know if it is for the right or the left to increase
#         #if trunk[0][0] < middle_img and trunk[1][0] < middle_img:
#         trunk[0][0] -= math.ceil(trunk_width / 4)
#         if trunk[0][0] < 0: # If the image cuts reach the left of the image
#             trunk[0][0] = 0
#         #else:
#         trunk[1][0] += math.ceil(trunk_width / 4)
#         if trunk[1][0] > width: # If the image cuts reach the right of the image
#             trunk[1][0] = width
#         update_trunks.append(trunk)
#
#     for trunk in update_trunks:
#         cropped_img = img[trunk[0][1]:trunk[1][1], trunk[0][0]:trunk[1][0]].copy()
#         canny_img = cv2.Canny(cropped_img, 150, 200)
#         # closed_img = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, SE, iterations=2)
#         dilated_img = cv2.dilate(canny_img, SE, iterations=2)
#         potential_trunks = find_potential_trunks(dilated_img, height)
#         if len(potential_trunks) > 0:
#             pot_trunk_area = 0
#             pot_trunk_index = 0
#             for i, pot_trunk in enumerate(potential_trunks):
#                 pot_trunk_height = pot_trunk[1][1] - pot_trunk[0][1]
#                 pot_trunk_width = pot_trunk[1][0] - pot_trunk[0][0]
#                 pot_trunk_area_aux = pot_trunk_height * pot_trunk_width
#                 if pot_trunk_area < pot_trunk_area_aux:
#                     pot_trunk_index = i
#             potential_trunks[pot_trunk_index][0][0] += trunk[0][0]
#             potential_trunks[pot_trunk_index][1][0] += trunk[0][0]
#             potential_trunks[pot_trunk_index][0][1] += trunk[0][1]
#             potential_trunks[pot_trunk_index][1][1] += trunk[0][1]
#             new_trunks += [potential_trunks[pot_trunk_index]]
#
#     return new_trunks
#
#
# # This function FINDS THE TRUNKS using the noise of the leaves and grass. Since the trunk doesn't have noise it's easier
# # to find it, by using histograms to see, vertically, where are "spikes" of black pixels. After taking the black stripes
# # of the image, the biggest continuous portion of horizontal blacks is defined as the trunk.
# def trunk_blacks(img):
#     # Define the structuring element using inbuilt CV2 function
#     SE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     dMat = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
#     height = img.shape[0]
#     width = img.shape[1]
#     middle_y = []
#     trunks_first_row = []
#     trunks_second_row = []
#
#     canny_img = cv2.Canny(img, 150, 200)
#     # cv2.imshow("canny_img", canny_img)
#
#     # For first row of trees
#     dilated_img1 = cv2.dilate(canny_img, SE, iterations=4)
#     potential_trunks = find_potential_trunks(dilated_img1, height)
#     trunks_first_row = middle_trunks(potential_trunks, height, [])
#
#     # For second row of trees
#     dilated_img2 = cv2.dilate(canny_img, SE, iterations=2)
#     if len(trunks_first_row) > 0:
#         dilated_img2, middle_y = clean_for_second_row(dilated_img2, trunks_first_row)
#     potential_trunks = find_potential_trunks(dilated_img2, height)
#     trunks_second_row = middle_trunks(potential_trunks, height, [])
#
#     trunks = trunks_first_row + trunks_second_row
#
#     return trunks
#
#
# # This function turns white known trees and are known to be sky or ground
# def clean_for_second_row(dilated_img, trunks):
#     middle_y = []
#     width = dilated_img.shape[1]
#     height = dilated_img.shape[0]
#     x1 = 0
#     y1 = height
#     x2 = width
#     y2 = 0
#     # Check if it was found one or more trees. If only one, then check if it is on the left side or the right side of the image
#     if len(trunks) > 0:
#         if len(trunks) >= 2:
#             for i in trunks:
#                 if i[1][0] < int(width/2):
#                     if i[1][0] > x1:
#                         x1 = i[1][0]+int((i[1][0]-i[0][0])/2)
#                 if i[0][0] > int(width/2):
#                     if i[0][0] < x2:
#                         x2 = i[0][0]-int((i[1][0]-i[0][0])/2)
#                 if i[0][1] < y1:
#                     y1 = i[0][1]
#                 if i[1][1] > y2:
#                     y2 = i[1][1]
#         else:
#             if trunks[0][0][0] > int(width/2):
#                 x2 = trunks[0][0][0]
#             else:
#                 x1 = trunks[0][1][0]
#             y1 = trunks[0][0][1]
#             y2 = trunks[0][1][1]
#
#         dilated_img[:y1,:] = [255]
#         dilated_img[y2:, :] = [255]
#         dilated_img[:, :x1] = [255]
#         dilated_img[:, x2:] = [255]
#         middle_y = [y1, y2]
#     return dilated_img, middle_y
#
#
# # Have to check this as equalizing and only using saturation is not really cleaning the image
# def clean_img(img):
#     img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#     # split the Hue, Saturation and Value color channels
#     hue, saturation, value = cv2.split(img_hsv)
#     cv2.imshow('Value Channel', value)
#     # value_equa = cv2.equalizeHist(value)
#     # cv2.imshow('Value equalized', value_equa)
#     return value
#
#
# # < summary >
# # Convert to mean
# # Direct access to memory - faster method
# # < /summary >
# # < param name="img" > image < /param >
# # < param name="imgCopy" > image < /param >
# def mean(img, imgCopy):
#     # direct access to the image memory(sequential)
#     # direction top left -> bottom right
#
#     width = len(img[0])
#     height = len(img)
#     nChan = len(img[0][0])  # number of channels - 3
#
#     imgCopy[0][0][0] = round(
#         (2 * (2 * img[0][0][0] + imgCopy[0][1][0]) + 2 * imgCopy[1][0][0] + imgCopy[1][1][0]) / 9.0)
#     imgCopy[0][0][1] = round(
#         (2 * (2 * img[0][0][1] + imgCopy[0][1][1]) + 2 * imgCopy[1][0][1] + imgCopy[1][1][1]) / 9.0)
#     imgCopy[0][0][2] = round(
#         (2 * (2 * img[0][0][2] + imgCopy[0][1][2]) + 2 * imgCopy[1][0][2] + imgCopy[1][1][2]) / 9.0)
#
#     x = 1
#     for x in range(width - 1):
#         imgCopy[0][x][0] = round((2 * (img[0][x - 1][0] + img[0][x][0] + imgCopy[0][x + 1][0]) + img[1][x - 1][0] +
#                                   imgCopy[1][x][0] + imgCopy[1][x + 1][0]) / 9.0)
#         imgCopy[0][x][1] = round((2 * (img[0][x - 1][1] + img[0][x][1] + imgCopy[0][x + 1][1]) + img[1][x - 1][1] +
#                                   imgCopy[1][x][1] + imgCopy[1][x + 1][1]) / 9.0)
#         imgCopy[0][x][2] = round((2 * (img[0][x - 1][2] + img[0][x][2] + imgCopy[0][x + 1][2]) + img[1][x - 1][2] +
#                                   imgCopy[1][x][2] + imgCopy[1][x + 1][2]) / 9.0)
#
#     imgCopy[0][width - 1][0] = round((2 * (2 * img[0][width - 1][0] + imgCopy[0][width - 2][0]) + 2 *
#                                       imgCopy[1][width - 1][0] + imgCopy[1][width - 2][0]) / 9.0)
#     imgCopy[0][width - 1][1] = round((2 * (2 * img[0][width - 1][1] + imgCopy[0][width - 2][1]) + 2 *
#                                       imgCopy[1][width - 1][1] + imgCopy[1][width - 2][1]) / 9.0)
#     imgCopy[0][width - 1][2] = round((2 * (2 * img[0][width - 1][2] + imgCopy[0][width - 2][2]) + 2 *
#                                       imgCopy[1][width - 1][2] + imgCopy[1][width - 2][2]) / 9.0)
#
#     y = 1
#     for y in range(height - 1):
#         imgCopy[y][0][0] = round((2 * (img[y - 1][0][0] + img[y][0][0] + imgCopy[y + 1][0][0]) + img[y - 1][1][0] +
#                                   imgCopy[y][1][0] + imgCopy[y + 1][1][0]) / 9.0)
#         imgCopy[y][0][1] = round((2 * (img[y - 1][0][1] + img[y][0][1] + imgCopy[y + 1][0][1]) + img[y - 1][1][1] +
#                                   imgCopy[y][1][1] + imgCopy[y + 1][1][1]) / 9.0)
#         imgCopy[y][0][2] = round((2 * (img[y - 1][0][2] + img[y][0][2] + imgCopy[y + 1][0][2]) + img[y - 1][1][2] +
#                                   imgCopy[y][1][2] + imgCopy[y + 1][1][2]) / 9.0)
#
#         x = 1
#         for x in range(width - 1):
#             # store in the image
#             imgCopy[y][x][0] = round((img[y - 1][x - 1][0] + img[y][x - 1][0] + imgCopy[y + 1][x - 1][0] +
#                                       img[y - 1][x][0] + img[y][x][0] + imgCopy[y + 1][x][0] + img[y - 1][x + 1][0] +
#                                       imgCopy[y][x + 1][0] + imgCopy[y + 1][x + 1][0]) / 9.0)
#             imgCopy[y][x][1] = round((img[y - 1][x - 1][1] + img[y][x - 1][1] + imgCopy[y + 1][x - 1][1] +
#                                       img[y - 1][x][1] + img[y][x][1] + imgCopy[y + 1][x][1] + img[y - 1][x + 1][1] +
#                                       imgCopy[y][x + 1][1] + imgCopy[y + 1][x + 1][1]) / 9.0)
#             imgCopy[y][x][2] = round((img[y - 1][x - 1][2] + img[y][x - 1][2] + imgCopy[y + 1][x - 1][2] +
#                                       img[y - 1][x][2] + img[y][x][2] + imgCopy[y + 1][x][2] + img[y - 1][x + 1][2] +
#                                       imgCopy[y][x + 1][2] + imgCopy[y + 1][x + 1][2]) / 9.0)
#
#         imgCopy[y][width - 1][0] = round((2 * (
#                     img[y - 1][width - 1][0] + img[y][width - 1][0] + imgCopy[y + 1][width - 1][0]) +
#                                           img[y - 1][width - 2][0] + imgCopy[y][width - 2][0] +
#                                           imgCopy[y + 1][width - 2][0]) / 9.0)
#         imgCopy[y][width - 1][1] = round((2 * (
#                     img[y - 1][width - 1][1] + img[y][width - 1][1] + imgCopy[y + 1][width - 1][1]) +
#                                           img[y - 1][width - 2][1] + imgCopy[y][width - 2][1] +
#                                           imgCopy[y + 1][width - 2][1]) / 9.0)
#         imgCopy[y][width - 1][2] = round((2 * (
#                     img[y - 1][width - 1][2] + img[y][width - 1][2] + imgCopy[y + 1][width - 1][2]) +
#                                           img[y - 1][width - 2][2] + imgCopy[y][width - 2][2] +
#                                           imgCopy[y + 1][width - 2][2]) / 9.0)
#
#     imgCopy[height - 1][0][0] = round((2 * (2 * img[height - 1][0][0] + imgCopy[height - 1][1][0]) + 2 *
#                                        imgCopy[height - 2][0][0] + imgCopy[height - 2][1][0]) / 9.0)
#     imgCopy[height - 1][0][1] = round((2 * (2 * img[height - 1][0][1] + imgCopy[height - 1][1][1]) + 2 *
#                                        imgCopy[height - 2][0][1] + imgCopy[height - 2][1][1]) / 9.0)
#     imgCopy[height - 1][0][2] = round((2 * (2 * img[height - 1][0][2] + imgCopy[height - 1][1][2]) + 2 *
#                                        imgCopy[height - 2][0][2] + imgCopy[height - 2][1][2]) / 9.0)
#
#     x = 1
#     for x in range(width - 1):
#         imgCopy[height - 1][x][0] = round((2 * (
#                     img[height - 1][x - 1][0] + img[height - 1][x][0] + imgCopy[height - 1][x + 1][0]) +
#                                            img[height - 2][x - 1][0] + imgCopy[height - 2][x][0] +
#                                            imgCopy[height - 2][x + 1][0]) / 9.0)
#         imgCopy[height - 1][x][1] = round((2 * (
#                     img[height - 1][x - 1][1] + img[height - 1][x][1] + imgCopy[height - 1][x + 1][1]) +
#                                            img[height - 2][x - 1][1] + imgCopy[height - 2][x][1] +
#                                            imgCopy[height - 2][x + 1][1]) / 9.0)
#         imgCopy[height - 1][x][2] = round((2 * (
#                     img[height - 1][x - 1][2] + img[height - 1][x][2] + imgCopy[height - 1][x + 1][2]) +
#                                            img[height - 2][x - 1][2] + imgCopy[height - 2][x][2] +
#                                            imgCopy[height - 2][x + 1][2]) / 9.0)
#
#     imgCopy[height - 1][width - 1][0] = round((2 * (
#                 2 * img[height - 1][width - 1][0] + imgCopy[height - 1][width - 2][0]) + 2 *
#                                                imgCopy[height - 2][width - 1][0] + imgCopy[height - 2][width - 2][
#                                                    0]) / 9.0)
#     imgCopy[height - 1][width - 1][1] = round((2 * (
#                 2 * img[height - 1][width - 1][1] + imgCopy[height - 1][width - 2][1]) + 2 *
#                                                imgCopy[height - 2][width - 1][1] + imgCopy[height - 2][width - 2][
#                                                    1]) / 9.0)
#     imgCopy[height - 1][width - 1][2] = round((2 * (
#                 2 * img[height - 1][width - 1][2] + imgCopy[height - 1][width - 2][2]) + 2 *
#                                                imgCopy[height - 2][width - 1][2] + imgCopy[height - 2][width - 2][
#                                                    2]) / 9.0)
#
#
# def squares_trees(img):
#     height, width = img.shape
#     k = 0
#     column_before_white = 0
#     thick_lines = []
#     # Divide the image in 3, because the middle is not used
#     side1_width = width/3
#     side2_width = 2*width/3
#     for j in range(width):
#         # Excluding the middle of the image
#         for i in range(height):
#             if (k % 2) == 0:
#                 if img[i][j] > 0:
#                     if column_before_white == 0:
#                         thick_lines.append((j, i))
#                         column_before_white = 1
#                     break
#             else:
#                 if img[i][j] > 0:
#                     if i == height - 1:
#                         if len(thick_lines) == k:
#                             thick_lines.append((j, i))
#                         else:
#                             thick_lines[k] = (j, i)
#                         column_before_white = 1
#                         break
#                     elif img[i + 1][j] == 0:
#                         if len(thick_lines) == k:
#                             thick_lines.append((j, i))
#                         else:
#                             thick_lines[k] = (j, i)
#                         column_before_white = 1
#                         break
#             if i == height-1:
#                 if column_before_white == 1:
#                     k += 1
#                 column_before_white = 0
#     return thick_lines
#
#
# def brown_mask(img):
#     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     brown_mask = cv2.inRange(hsv_img, (10, 50, 0), (29, 255, 255))
#     # brown_mask = cv2.bitwise_not(brown_mask_not)
#     brown_mask_img = cv2.bitwise_and(img, img, mask=brown_mask)
#     # cv2.imshow("HSVB1", brown_mask_img)
#     cv2.dilate(brown_mask_img, (3, 3), brown_mask_img)
#     # cv2.imshow("HSVB2", brown_mask_img)
#     cv2.dilate(brown_mask_img, (1, 3), brown_mask_img)
#     # cv2.imshow("HSVB3", brown_mask_img)
#     cv2.dilate(brown_mask_img, (1, 3), brown_mask_img)
#     # cv2.imshow("HSVB4", brown_mask_img)
#     cv2.erode(brown_mask_img, (3, 3), brown_mask_img)
#     cv2.imshow("HSV_Brown", brown_mask_img)
#     # brown_mask_img[np.where((brown_mask_img == [0, 0, 0]).all(axis=2))] = [255, 0, 0]
#     # cv2.imshow("HSVB", brown_mask_img)
#     '''
#     brown_mask1_not = cv2.inRange(hsv_img, (10, 50, 0), (29, 255, 170))
#     brown_mask1 = cv2.bitwise_not(brown_mask1_not)
#     brown_mask1_img = cv2.bitwise_and(img, img, mask=brown_mask1)
#     brown_mask1_img[np.where((brown_mask1_img == [0, 0, 0]).all(axis=2))] = [255, 0, 0]
#     cv2.imshow("HSVB1", brown_mask1_img)
#     '''
#
#     return brown_mask_img
#
#
# def green_mask(img):
#     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     green_mask = cv2.inRange(hsv_img, (31, 0, 0), (75, 255, 255))
#     # green_mask = cv2.bitwise_not(green_mask_not)
#     green_mask_img = cv2.bitwise_and(img, img, mask=green_mask)
#     # cv2.imshow("HSVG1", green_mask_img)
#     cv2.dilate(green_mask_img, (3, 3), green_mask_img)
#     cv2.dilate(green_mask_img, (3, 3), green_mask_img)
#     # cv2.imshow("HSVG2", green_mask_img)
#     cv2.erode(green_mask_img, (3, 3), green_mask_img)
#     cv2.imshow("HSV_Green", green_mask_img)
#     # green_mask_img[np.where((green_mask_img == [0, 0, 0]).all(axis=2))] = [0, 0, 255]
#     # cv2.imshow("HSVG", green_mask_img)
#     '''
#     green_mask1_not = cv2.inRange(hsv_img, (31, 0, 0), (75, 255, 190))
#     green_mask1 = cv2.bitwise_not(green_mask1_not)
#     green_mask1_img = cv2.bitwise_and(img, img, mask=green_mask1)
#     green_mask1_img[np.where((green_mask1_img == [0, 0, 0]).all(axis=2))] = [0, 0, 255]
#     cv2.imshow("HSVG1", green_mask1_img)
#     '''
#     return green_mask_img
#
#
# def green_and_brown_mask_combined(img):
#     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     green_mask_not = cv2.inRange(hsv_img, (31, 0, 0), (75, 255, 255))
#     green_mask = cv2.bitwise_not(green_mask_not)
#     green_mask_img = cv2.bitwise_and(img, img, mask=green_mask)
#     cv2.imshow("HSVG1", green_mask_img)
#     cv2.dilate(green_mask_img, (3, 3), green_mask_img)
#     cv2.dilate(green_mask_img, (3, 3), green_mask_img)
#     cv2.imshow("HSVG2", green_mask_img)
#     cv2.erode(green_mask_img, (3, 3), green_mask_img)
#     cv2.imshow("HSVG3", green_mask_img)
#     green_mask_img[np.where((green_mask_img == [0, 0, 0]).all(axis=2))] = [0, 0, 255]
#     # cv2.imshow("HSVG", green_mask_img)
#     '''
#     green_mask1_not = cv2.inRange(hsv_img, (31, 0, 0), (75, 255, 190))
#     green_mask1 = cv2.bitwise_not(green_mask1_not)
#     green_mask1_img = cv2.bitwise_and(img, img, mask=green_mask1)
#     green_mask1_img[np.where((green_mask1_img == [0, 0, 0]).all(axis=2))] = [0, 0, 255]
#     cv2.imshow("HSVG1", green_mask1_img)
#     '''
#
#     brown_mask_not = cv2.inRange(hsv_img, (10, 50, 0), (29, 255, 255))
#     brown_mask = cv2.bitwise_not(brown_mask_not)
#     brown_mask_img = cv2.bitwise_and(img, img, mask=brown_mask)
#     cv2.imshow("HSVB1", brown_mask_img)
#     cv2.dilate(brown_mask_img, (3, 3), brown_mask_img)
#     cv2.dilate(brown_mask_img, (3, 3), brown_mask_img)
#     cv2.imshow("HSVB2", brown_mask_img)
#     cv2.erode(brown_mask_img, (3, 3), brown_mask_img)
#     cv2.imshow("HSVB3", brown_mask_img)
#     brown_mask_img[np.where((brown_mask_img == [0, 0, 0]).all(axis=2))] = [255, 0, 0]
#     # cv2.imshow("HSVB", brown_mask_img)
#     '''
#     brown_mask1_not = cv2.inRange(hsv_img, (10, 50, 0), (29, 255, 170))
#     brown_mask1 = cv2.bitwise_not(brown_mask1_not)
#     brown_mask1_img = cv2.bitwise_and(img, img, mask=brown_mask1)
#     brown_mask1_img[np.where((brown_mask1_img == [0, 0, 0]).all(axis=2))] = [255, 0, 0]
#     cv2.imshow("HSVB1", brown_mask1_img)
#     '''
#
#     paint_img = cv2.bitwise_and(brown_mask_img, green_mask_img)
#     cv2.imshow("PAINT", paint_img)
#
#     return paint_img
#
#
# def green_brown_mask(img):
#     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     green_mask = cv2.inRange(hsv_img, (31, 0, 0), (75, 255, 255))
#     green_mask_img = cv2.bitwise_and(img, img, mask=green_mask)
#     cv2.dilate(green_mask_img, (3, 3), green_mask_img)
#     cv2.dilate(green_mask_img, (3, 3), green_mask_img)
#     cv2.erode(green_mask_img, (3, 3), green_mask_img)
#     cv2.imshow("HSV_Green", green_mask_img)
#     return green_mask_img
#
#
# def lines_by_hough(img, org_img, minlength):
#     lines = cv2.HoughLinesP(img, 1, np.pi, 100, minLineLength=minlength, maxLineGap=5)
#     if lines is not None:
#         for i in range(0, len(lines)):
#             pt1 = (lines[i][0][0], lines[i][0][1])
#             pt2 = (lines[i][0][2], lines[i][0][3])
#             cv2.line(org_img, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)
#
#     return org_img
#
#
# def vertical_sobel(img):
#     grad_x = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
#     grad_y = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
#     abs_grad_x = cv2.convertScaleAbs(grad_x)
#     abs_grad_y = cv2.convertScaleAbs(grad_y)
#     grad = cv2.addWeighted(abs_grad_x, 1, abs_grad_y, 0, 0)
#     # (T, sob_img) = cv2.threshold(grad, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
#     # cv2.imshow("sobel_grad_img", grad)
#
#     return grad
#
#
# def horizontal_sobel(img):
#     grad_x = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
#     grad_y = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
#     abs_grad_x = cv2.convertScaleAbs(grad_x)
#     abs_grad_y = cv2.convertScaleAbs(grad_y)
#     grad = cv2.addWeighted(abs_grad_x, 0, abs_grad_y, 1, 0)
#     # (T, sob_img) = cv2.threshold(grad, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
#     # cv2.imshow("sobel_grad_img", grad)
#
#     return grad
#
#
# def sobel(img):
#     grad_x = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
#     grad_y = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
#     abs_grad_x = cv2.convertScaleAbs(grad_x)
#     abs_grad_y = cv2.convertScaleAbs(grad_y)
#     grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
#     # (T, sob_img) = cv2.threshold(grad, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
#     # cv2.imshow("sobel_grad_img", grad)
#
#     return grad
#
#
# def template_match(img):
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     template = cv2.imread('C:\\Users\\afons\\Documents\\Tese\\Projeto_notNN\\data\\WalnutAcresTree.png', cv2.IMREAD_GRAYSCALE)
#     w, h = template.shape[::-1]
#
#     res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#     top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     cv2.rectangle(img_gray, top_left, bottom_right, (0, 0, 255), 1)
#     cv2.imshow('res', img_gray)
#
#
# # TRIED TO GO BY COLOR
# # NOT WORKING VERY GOOD
# # probably not working, haven't tried it in a while
# def using_colors(img):
#     height, width, _ = img.shape
#     # resized_size = (int(width / 2), int(height / 2))
#     # img = cv2.resize(img1, resized_size, interpolation=cv2.INTER_AREA)
#     cv2.imshow("thumbnail", img)
#
#     blur_img = cv2.GaussianBlur(img, (7, 7), 0)
#     # cv2.imshow("blur", blur_img)
#
#     brown_img = brown_mask(blur_img)
#     # cv2.imshow("brown", brown_img)
#
#     # paint_img = green_and_brown_mask(blur_img)
#
#     gray_img = cv2.cvtColor(brown_img, cv2.COLOR_BGR2GRAY)
#     # cv2.imshow("gray_B", gray_img)
#
#     # des = cv2.bitwise_not(gray_img)
#     # cv2.imshow("BIT NOT", des)
#     contour, hier = cv2.findContours(gray_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#
#     for cnt in contour:
#         cv2.drawContours(gray_img, [cnt], 0, 255, -1)
#
#     # gray = cv2.bitwise_not(gray_img)
#     # cv2.imshow("draw_contours_B", gray_img)
#
#     vertical_sobel_img = vertical_sobel(gray_img)
#     # cv2.imshow("vertical_sobel_B", vertical_sobel_img)
#
#     hough_lines = lines_by_hough(vertical_sobel_img, img, 100)
#     # cv2.imshow("Hough_Lines_B", hough_lines)
#
#     canny_img = cv2.Canny(gray_img, 100, 50)
#     # cv2.imshow("canny_B", canny_img)
#
#     # GREEN
#     # BETTER TO USE A BIGGER KERNEL LIKE [9,9] TO THE GREENS
#     blur_img = cv2.GaussianBlur(img, (3, 3), 0)
#     # cv2.imshow("blur_9", blur_img)
#
#     green_img = green_mask(blur_img)
#
#     gray_img = cv2.cvtColor(green_img, cv2.COLOR_BGR2GRAY)
#     # cv2.imshow("gray_G", gray_img)
#
#     # des = cv2.bitwise_not(gray_img)
#     # cv2.imshow("BIT NOT", des)
#     contour, hier = cv2.findContours(gray_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#
#     for cnt in contour:
#         cv2.drawContours(gray_img, [cnt], 0, 255, -1)
#
#     # gray = cv2.bitwise_not(gray_img)
#     # cv2.imshow("draw_contours_G", gray_img)
#
#     vertical_sobel_img = vertical_sobel(gray_img)
#     # cv2.imshow("vertical_sobel_G", vertical_sobel_img)
#
#     hough_lines = lines_by_hough(vertical_sobel_img, img, 100)
#     # cv2.imshow("Hough_Lines_G", hough_lines)
#
#     canny_img = cv2.Canny(gray_img, 100, 50)
#     # cv2.imshow("canny_G", canny_img)
#
#
# # SEE 30/11/2023 NOTES; BLUR AND REMOVAL OF DETAIL, SO ONLY THE STRONGER CONTRASTS WERE SHOWN
# def trunk_contrast_canny(img):
#     dMat = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
#     height, width, _ = img.shape
#     resized_size = (int(width / 1), int(height / 1))
#     img = cv2.resize(img, resized_size, interpolation=cv2.INTER_AREA)
#     cv2.imshow("thumbnail", img)
#
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("gray_img_1", gray_img)
#
#     blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
#     cv2.imshow("blur", blur_img)
#
#     canny_img = cv2.Canny(blur_img, 50, 200)
#     cv2.imshow("canny_img", canny_img)
#
#     dilated_img = cv2.dilate(canny_img, dMat, iterations=4)
#     cv2.imshow("dilated_img", dilated_img)
#
#     # Get a completely black img to draw the hough lines
#     (thresh, black_img) = cv2.threshold(gray_img, 255, 255, cv2.THRESH_BINARY)
#     hough_lines_canny = lines_by_hough(dilated_img, black_img, int(height / 2))
#     cv2.imshow("Hough_Lines_c", hough_lines_canny)
#
#     thick_lines = squares_trees(hough_lines_canny)
#
#     k = 0
#     # Draw each rectangle on the resized image
#     while k < len(thick_lines):
#         if len(thick_lines) % 2 != 0 and k == len(thick_lines) - 1:
#             break
#         k += 1
#         cv2.rectangle(img, thick_lines[k - 1], thick_lines[k], (0, 0, 255), 3)
#         k += 1
#
#     cv2.imshow("Final image", img)
#
#
# # CANNY IS BETTER THAN VERTICAL SOBEL (have to check again after talking to professor)
# def trunk_contrast_vertical_sobel(img):
#     dMat = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
#     # resized_size = (int(img.shape[0] / 2), int(img.shape[1] / 2))
#     # img = cv2.resize(img, resized_size, interpolation=cv2.INTER_AREA)
#
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("gray_img_1", gray_img)
#
#     vertical_sobel_img = vertical_sobel(gray_img)
#     cv2.imshow("vertical_sobel", vertical_sobel_img)
#
#     (T1, thresh_img) = cv2.threshold(vertical_sobel_img, 127, 255, cv2.THRESH_BINARY)
#     cv2.imshow("sobel_grad_img", thresh_img)
#
#     dMat = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
#     dilated_img_vs = cv2.dilate(thresh_img, dMat, iterations=1)
#     cv2.imshow("dilated_img_vs", dilated_img_vs)
#
#     # Get a completely black img to draw the hough lines
#     (thresh, black_img) = cv2.threshold(gray_img, 255, 255, cv2.THRESH_BINARY)
#     hough_lines = lines_by_hough(dilated_img_vs, black_img, int(img.shape[0] / 4))
#     cv2.imshow("Hough_Lines_c", hough_lines)
#
#     thick_lines = squares_trees(hough_lines)
#
#     k = 0
#     # Draw each rectangle on the resized image
#     while k < len(thick_lines):
#         if len(thick_lines) % 2 != 0 and k == len(thick_lines) - 1:
#             break
#         k += 1
#         cv2.rectangle(img, thick_lines[k - 1], thick_lines[k], (0, 0, 255), 3)
#         k += 1
#
#     cv2.imshow("Final image", img)
#
#
# def trunk_contrast_horizontal_sobel(img):
#     dMat = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
#     # resized_size = (int(img.shape[0] / 2), int(img.shape[1] / 2))
#     # img = cv2.resize(img, resized_size, interpolation=cv2.INTER_AREA)
#
#     blur_img = cv2.GaussianBlur(img, (1, 9), 0)
#     cv2.imshow("blur_img", blur_img)
#
#     # gray_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
#     # cv2.imshow("gray_img_1", gray_img)
#     # horizontal_sobel_img = horizontal_sobel(gray_img)
#     # cv2.imshow("horizontal sobel", horizontal_sobel_img)
#     # (T1, thresh_img) = cv2.threshold(horizontal_sobel_img, 150, 255, cv2.THRESH_BINARY)
#     # cv2.imshow("sobel_grad_img", thresh_img)
#
#     canny_img = cv2.Canny(blur_img, 190, 230)
#     cv2.imshow("canny_img", canny_img)
#
#     # # Define a black  mask
#     # mask = np.zeros_like(canny_img)
#     # # Flip the mask color to white
#     # mask[:] = [255]
#     half_width = int(img.shape[1] / 2)
#     # mask_left_side = cv2.rectangle(mask, (0,0), (half_width,img.shape[0]), (0,0,0))
#     # left_side = cv2.bitwise_and(canny_img, mask_left_side)
#     # cv2.imshow("left side", left_side)
#
#     cdstP = np.copy(img)
#
#     linesP = cv2.HoughLinesP(canny_img, 1, np.pi / 180, 50, None, 100, 30)
#     if linesP is not None:
#         for i in range(0, len(linesP)):
#             l = linesP[i][0]
#             cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv2.LINE_AA)
#
#     cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
#
#     # # Find black pixels per column
#     # col_counts = cv2.reduce(canny_img, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
#     # plt.plot(col_counts)
#     # # plt.show()
#     # col_counts = np.transpose(col_counts)
#     # col_counts = col_counts[0]
#     # col_counts_max = max(col_counts)
#     # horizon_y = np.where(col_counts == col_counts_max)
#     half_height = int(img.shape[0]/2)
#     cv2.line(img, (0,  half_height), (img.shape[1], half_height), (255, 255, 0))
#     cv2.line(img, (half_width, 0), (half_width, img.shape[0]), (255, 0, 0))
#     cv2.imshow("horizon line", img)
#
#
# def fill_spots5(img):
#     copied_img = img.copy()
#     pass_value = 15*255
#     for y in range(img.shape[0]):
#         for x, value in enumerate(img[y]):
#             if value == 0:
#                 try: kernel_value = np.sum(img[y-2:y+3,x-2:x+3])
#                 except:
#                     continue
#                 if kernel_value >= pass_value:
#                     copied_img[y,x] = 255
#     return copied_img
#
#
# def fill_spots3(img):
#     copied_img = img.copy()
#     pass_value = 5*255
#     for y in range(img.shape[0]):
#         for x, value in enumerate(img[y]):
#             if value == 0:
#                 try: kernel_value = np.sum(img[y-1:y+2,x-1:x+2])
#                 except:
#                     continue
#                 if kernel_value >= pass_value:
#                     copied_img[y,x] = 255
#     return copied_img
#
#
# def colorQuant(img, Z, K, criteria):
#     ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#
#     # Now convert back into uint8, and make original image
#     center = np.uint8(center)
#     res = center[label.flatten()]
#     res2 = res.reshape((img.shape))
#     return res2
#
#
# def quantization(img):
#     # reshape the image into a feature vector so that k-means can be applied
#     Z = img.reshape((-1, 3))
#
#     # convert to np.float32
#     Z = np.float32(Z)
#
#     # define criteria, number of clusters(K) and apply kmeans()
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
#
#     res1 = colorQuant(img, Z, 8, criteria)
#     cv2.imshow("res1", res1)
#     return res1
#
# """
# max_value = 255
# max_value_H = 360//2
# low_H = 0
# low_S = 0
# low_V = 0
# high_H = max_value_H
# high_S = max_value
# high_V = max_value
# window_capture_name = 'Video Capture'
# window_detection_name = 'Object Detection'
# low_H_name = 'Low H'
# low_S_name = 'Low S'
# low_V_name = 'Low V'
# high_H_name = 'High H'
# high_S_name = 'High S'
# high_V_name = 'High V'
# def on_low_H_thresh_trackbar(val):
#     global low_H
#     global high_H
#     low_H = val
#     low_H = min(high_H-1, low_H)
#     cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
# def on_high_H_thresh_trackbar(val):
#     global low_H
#     global high_H
#     high_H = val
#     high_H = max(high_H, low_H+1)
#     cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
# def on_low_S_thresh_trackbar(val):
#     global low_S
#     global high_S
#     low_S = val
#     low_S = min(high_S-1, low_S)
#     cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
# def on_high_S_thresh_trackbar(val):
#     global low_S
#     global high_S
#     high_S = val
#     high_S = max(high_S, low_S+1)
#     cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
# def on_low_V_thresh_trackbar(val):
#     global low_V
#     global high_V
#     low_V = val
#     low_V = min(high_V-1, low_V)
#     cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
# def on_high_V_thresh_trackbar(val):
#     global low_V
#     global high_V
#     high_V = val
#     high_V = max(high_V, low_V+1)
#     cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)
#
#
# cv2.namedWindow(window_capture_name)
# cv2.namedWindow(window_detection_name)
# cv2.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
# cv2.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
# cv2.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
# cv2.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
# cv2.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
# cv2.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
# """
#
# def trunk_blacks_blobs(img):
#     dMat = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
#     height = img.shape[0]
#     width = img.shape[1]
#     while True:
#         frame_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
#
#         cv2.imshow(window_capture_name, img)
#         cv2.imshow(window_detection_name, frame_threshold)
#
#         key = cv2.waitKey(30)
#         if key == ord('q') or key == 27:
#             break
#
#     """
#     #frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     #frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
#
#     canny_img = cv2.Canny(img, 150, 200)
#     cv2.imshow("canny_img", canny_img)
#
#     dilated_img = cv2.dilate(canny_img, dMat, iterations=1)
#     cv2.imshow("dilated_img", dilated_img)
#
#     #filled_img = fill_spots(dilated_img)
#     #cv2.imshow("filled img", filled_img)
#
#     # Apply Laplacian of Gaussian
#     \"""
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # cv2.imshow("gray_img_1", gray_img)
#     blobs_log = cv2.Laplacian(gray_img, cv2.CV_64F)
#     blobs_log = np.uint8(np.absolute(blobs_log))
#     cv2.imshow("laplacian", blobs_log)
#     (T1, thresh_img) = cv2.threshold(blobs_log, 30, 255, cv2.THRESH_BINARY)
#     cv2.imshow("black and white", thresh_img)
#     # Define the structuring element using inbuilt CV2 function
#     SE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     dilated_img1 = cv2.dilate(thresh_img, SE, iterations=2)
#     cv2.imshow("dilated_img1", dilated_img1)
#     eroded_img1 = cv2.erode(dilated_img1, SE, iterations=2)
#     cv2.imshow("eroded_img1", eroded_img1)
#     \"""
#
#     # Find black pixels per column
#     mask = np.uint8(np.where(dilated_img == 0, 1, 0))  # THIS MAY NOT BE NEEDED
#     col_counts = cv2.reduce(mask, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
#
#     i = 3
#     k = 0
#     col_spikes = []
#     min_spike = int(col_counts.max() * 0.20)
#     # Find the spikes
#     while i < len(col_counts[0]) - 3:
#         # Going up
#         if col_counts[0][i] - col_counts[0][i + 3] <= -min_spike:
#             if k == len(col_spikes):
#                 col_spikes.append(i)
#             else:
#                 col_spikes[k] = i
#         # Going Down
#         if col_counts[0][i - 3] - col_counts[0][i] >= min_spike and len(col_spikes) > k:
#             # widens the area found
#             trunk_width = i - col_spikes[k]
#             if col_spikes[k] - trunk_width > 0:
#                 col_spikes[k] -= trunk_width
#             else:
#                 col_spikes[k] = 0
#             if i + trunk_width < len(col_counts[0]):
#                 col_spikes.append(i + trunk_width)
#             else:
#                 col_spikes.append(len(col_counts[0]))
#             k += 2
#
#         i += 1
#
#     plt.plot(col_counts[0])
#     # plt.show()
#     plt.plot(col_counts[0])
#     for i in col_spikes:
#         plt.axvline(i, color='r')
#     # plt.show()
#
#     if len(col_spikes) % 2 == 1:
#         col_spikes.pop()
#
#     # Find the trunks in height on the "cut" image
#     i = 0
#     potential_trunks = []
#     while i < len(col_spikes):
#         # cv2.imshow("Image cut {}".format(i / 2), dilated_img[:, col_spikes[i]:col_spikes[i + 1]])
#
#         # Count black pixels per row after separating the image
#         row_counts = cv2.reduce(mask[:, col_spikes[i]:col_spikes[i + 1]], 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
#
#         top_trunk = 0
#         bottom_trunk = 0
#         top_trunk_temp = -1
#         for j, k in enumerate(row_counts):
#             if k == 0 or j == 0 or j == len(row_counts) - 1:
#                 if top_trunk_temp == -1:
#                     top_trunk_temp = j
#                 else:
#                     if j - top_trunk_temp > bottom_trunk - top_trunk:
#                         top_trunk = top_trunk_temp
#                         bottom_trunk = j
#                     top_trunk_temp = -1
#                     if j < len(row_counts) - 1 and row_counts[j + 1] > 0:
#                         top_trunk_temp = j
#
#         plt.plot(row_counts)
#         plt.axvline(bottom_trunk, color='r')
#         plt.axvline(top_trunk, color='r')
#         plt.title("Image cut {}".format(i / 2))
#         # plt.show()
#
#         cv2.rectangle(img, (col_spikes[i], top_trunk), (col_spikes[i + 1], bottom_trunk), (0, 0, 255), 1)
#         potential_trunks.append([[col_spikes[i], top_trunk], [col_spikes[i + 1], bottom_trunk]])
#
#         i += 2  # in order to jump to the next "pair"
#     # cv2.imshow("Image with rectangles", img)
#
#     # Find the trunks in height on the "cut" image
#     i = 0
#     trunks = middle_trunks(potential_trunks, height)
#     while i < len(trunks):
#         cv2.rectangle(img, trunks[i][0], trunks[i][1], (0, 255, 0), 1)
#         i += 1  # in order to jump to the next "pair"
#     cv2.imshow("Only middle rectangles", img)
#     """
#
