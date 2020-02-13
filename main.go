package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"strings"
	"sync"
)

func main() {
	urls := []string{"https://www.bienmodel.net:5000/api/v1/date-set/109/train.md5",
		"https://www.bienmodel.net:5000/api/v1/date-set/109/update_new_columns_trains_sets.z01",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/update_new_columns_trains_sets.z02",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/update_new_columns_trains_sets.z03",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/update_new_columns_trains_sets.z04",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/update_new_columns_trains_sets.z05",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/update_new_columns_trains_sets.z06",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/update_new_columns_trains_sets.z07",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/update_new_columns_trains_sets.z08",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/update_new_columns_trains_sets.z09",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/update_new_columns_trains_sets.z10",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/update_new_columns_trains_sets.z11",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/update_new_columns_trains_sets.z12",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/update_new_columns_trains_sets.z13",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/update_new_columns_trains_sets.z14",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/update_new_columns_trains_sets.z15",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/update_new_columns_trains_sets.z16",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/update_new_columns_trains_sets.z17",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/update_new_columns_trains_sets.zip",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/val_labels_v1.csv",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/val_sets_v1.z01",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/val_sets_v1.z02",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/val_sets_v1.z03",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/val_sets_v1.z04",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/val_sets_v1.zip",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/delta_validation_set.csv",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/delta_validation_label.csv",

		"https://www.bienmodel.net:5000/api/v1/date-set/109/val.md5",
	}

	ch := make(chan bool, 4)

	rootDir := "C:/Users/User/Desktop/go_download/"
	if _, err := os.Open(rootDir); err != nil {
		err = os.MkdirAll(rootDir, os.ModePerm)
		if err != nil {
			fmt.Println(err)
		}
	}
	var wg sync.WaitGroup
	wg.Add(len(urls))
	for _, url := range urls {
		components := strings.Split(url, "/")
		outName := components[len(components)-1]
		absDir := path.Join(rootDir, outName)
		go getUrl(url, absDir, ch, &wg)
	}
	wg.Wait()
}

func getUrl(url, outName string, ch chan bool, wg *sync.WaitGroup) {
	ch <- true
	defer func() {
		fmt.Printf("finishing %q\n", outName)
		<-ch
		wg.Done()
	}()
	resp, err := http.Get(url)
	if err != nil {
		fmt.Println(err)
	}

	defer resp.Body.Close()

	f, err := os.OpenFile(outName, os.O_WRONLY|os.O_CREATE, os.ModePerm)
	if err != nil {
		fmt.Println(err)
	}

	defer f.Close()

	_, err = io.Copy(f, resp.Body)

	if err != nil {
		fmt.Println(err)
	}
}
