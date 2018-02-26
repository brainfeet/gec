(defproject gec "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url  "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [aid "0.1.1"]
                 [cheshire "5.8.0"]
                 [clj-time "0.14.2"]
                 [com.rpl/specter "1.0.5"]
                 [funcool/cats "2.1.0"]
                 [me.raynes/fs "1.4.6"]]
  :plugins [[lein-auto "0.1.3"]
            [lein-ancient "0.6.10"]]
  :main ^:skip-aot gec.core
  :target-path "target/%s"
  :auto {"rsync" {:paths        ["."]
                  :file-pattern #"\.(clj|json|py)$"}}
  :profiles {:uberjar {:aot :all}}
  :aliases {"rsync" ["run" "rsync"]})
