(defproject gec "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url  "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [aid "0.1.1"]
                 [cheshire "5.8.0"]
                 [com.rpl/specter "1.0.5"]
                 [funcool/cats "2.1.0"]
                 [me.raynes/fs "1.4.6"]]
  :main ^:skip-aot gec.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
