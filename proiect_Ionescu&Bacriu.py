import cv2
import numpy as np


def apply_median_filter(src):
    dst = src.copy()
    rows, cols, _ = src.shape
    padded_src = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    for i in range(rows):
        for j in range(cols):
            window = padded_src[i:i + 3, j:j + 3]
            dst[i, j, 0] = np.median(window[:, :, 0])
            dst[i, j, 1] = np.median(window[:, :, 1])
            dst[i, j, 2] = np.median(window[:, :, 2])
    return dst


def apply_mean_filter(src):
    dst = src.copy()
    rows, cols, _ = src.shape
    padded_src = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    for i in range(rows):
        for j in range(cols):
            window = padded_src[i:i + 3, j:j + 3]
            dst[i, j, 0] = np.mean(window[:, :, 0])
            dst[i, j, 1] = np.mean(window[:, :, 1])
            dst[i, j, 2] = np.mean(window[:, :, 2])
    return dst


def apply_gaussian_filter(src):
    dst = src.copy()
    rows, cols, _ = src.shape
    padded_src = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])
    kernel_sum = 16.0

    for i in range(rows):
        for j in range(cols):
            window = padded_src[i:i + 3, j:j + 3]
            dst[i, j, 0] = np.sum(window[:, :, 0] * kernel) / kernel_sum
            dst[i, j, 1] = np.sum(window[:, :, 1] * kernel) / kernel_sum
            dst[i, j, 2] = np.sum(window[:, :, 2] * kernel) / kernel_sum
    return dst


def apply_min_filter(src):
    dst = src.copy()
    rows, cols, _ = src.shape
    padded_src = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    for i in range(rows):
        for j in range(cols):
            window = padded_src[i:i + 3, j:j + 3]
            dst[i, j, 0] = np.min(window[:, :, 0])
            dst[i, j, 1] = np.min(window[:, :, 1])
            dst[i, j, 2] = np.min(window[:, :, 2])
    return dst


def apply_selective_median_filter(src):
    dst = src.copy()
    rows, cols, _ = src.shape
    padded_src = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    for i in range(rows):
        for j in range(cols):
            b, g, r = src[i, j]
            is_black_noise = (b < 10 and g < 10 and r < 10)
            is_white_noise = (b > 245 and g > 245 and r > 245)

            if is_black_noise or is_white_noise:
                window = padded_src[i:i + 3, j:j + 3]
                dst[i, j, 0] = np.median(window[:, :, 0])
                dst[i, j, 1] = np.median(window[:, :, 1])
                dst[i, j, 2] = np.median(window[:, :, 2])
    return dst


def apply_manual_inpaint(src, iterations=3):
    dst = src.copy()
    rows, cols, _ = src.shape

    for pas in range(iterations):
        padded_dst = cv2.copyMakeBorder(dst, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

        for i in range(rows):
            for j in range(cols):
                b, g, r = dst[i, j]

                if b > 200 and g > 200 and r > 200:
                    window = padded_dst[i:i + 3, j:j + 3]

                    valid_b, valid_g, valid_r = [], [], []
                    for wi in range(3):
                        for wj in range(3):
                            wb, wg, wr = window[wi, wj]
                            if not (wb > 200 and wg > 200 and wr > 200):
                                valid_b.append(wb)
                                valid_g.append(wg)
                                valid_r.append(wr)

                    if len(valid_b) > 0:
                        dst[i, j, 0] = int(sum(valid_b) / len(valid_b))
                        dst[i, j, 1] = int(sum(valid_g) / len(valid_g))
                        dst[i, j, 2] = int(sum(valid_r) / len(valid_r))
    return dst


def main():
    choice = 0
    while choice != -1:
        print("\n--- Proiect: Restaurarea Imaginilor ---")
        print("1. Filtru Median")
        print("2. Filtru de Mediere")
        print("3. Filtru Gaussian")
        print("4. Filtru de Minim")
        print("5. Filtru Median Selectiv")
        print("6. Inpainting Manual")
        print("-1. Iesire din program")

        try:
            choice = int(input("Optiunea ta (1-6 sau -1): "))
        except ValueError:
            print("Eroare: Introdu un numar valid!")
            continue

        if choice == -1:
            break

        image_paths = {
            1: "test_1_median.bmp",
            2: "test_2_mediere.bmp",
            3: "test_3_gaussian.bmp",
            4: "test_4_minim.bmp",
            5: "test_5_selectiv.bmp",
            6: "test_6_inpaint.bmp"
        }

        if choice not in image_paths:
            print("Optiune invalida! Alege un numar intre 1 si 6.")
            continue

        image_path = image_paths[choice]
        src = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if src is None:
            print(f"Eroare: Nu s-a putut incarca imaginea '{image_path}'!")
            continue

        print(f"Se proceseaza imaginea '{image_path}'... (asteapta 1-3 secunde)")

        if choice == 1:
            dst = apply_median_filter(src)
        elif choice == 2:
            dst = apply_mean_filter(src)
        elif choice == 3:
            dst = apply_gaussian_filter(src)
        elif choice == 4:
            dst = apply_min_filter(src)
        elif choice == 5:
            dst = apply_selective_median_filter(src)
        elif choice == 6:
            dst = apply_manual_inpaint(src)

        cv2.imshow(f"Originala ({image_path})", src)
        cv2.imshow("Rezultat Restaurare", dst)

        cv2.imwrite("restaurat.bmp", dst)
        print("Imaginea a fost salvata ca 'restaurat.bmp'.")

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()